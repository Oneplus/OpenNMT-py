#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import argparse
import random
import onmt
import onmt.IO
import opts
import torch
from torch.autograd import Variable
import glob
import itertools

parser = argparse.ArgumentParser(description='ensemble.py')
opts.add_md_help_argument(parser)
opts.generate_opts(parser)

opt = parser.parse_args()
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.gpu > -1:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


def get_batch_stats(batch, tgt, bos_id, eos_id):
    stats = onmt.Statistics()
    gold = batch.tgt.data.clone()
    pred = tgt.clone()

    # TODO: it's counted in wrong way.
    pred_mask = (pred != bos_id) * (pred != eos_id)
    stats.n_words = pred_mask.sum()
    if gold.size(0) < pred.size(0):
        pred = pred.narrow(0, 0, gold.size(0))
        pred_mask = pred_mask.narrow(0, 0, gold.size(0))
    elif pred.size(0) < gold.size(0):
        gold = gold.narrow(0, 0, pred.size(0))
    matched = (pred == gold) * pred_mask
    stats.n_correct = matched.sum()
    return stats


def load_translators():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    translators = []
    for model in glob.glob(opt.models):
        # tricky, override the model path.
        opt.model = model
        translators.append(onmt.Translator(opt, dummy_opt.__dict__))
    n_translators = len(translators)
    assert n_translators > 0
    print('Created {0} translators.'.format(n_translators))
    return translators


def generate():
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    tt = torch.cuda if opt.cuda else torch

    translators = load_translators()

    test_data = torch.load(opt.data + '.pt')
    for example in test_data.examples:
        if 'selected_indices' in example.__dict__:
            del example.selected_indices
        if 'selected_distrib' in example.__dict__:
            del example.selected_distrib
    fields = onmt.IO.ONMTDataset.load_fields(torch.load(opt.vocab + '.pt'))
    fields = dict([(k, f) for (k, f) in fields.items() if k in test_data.examples[0].__dict__])

    test_data.fields = fields
    test_iter = onmt.IO.OrderedIterator(
        dataset=test_data, batch_size=opt.batch_size,
        device=opt.gpu,
        train=False, sort=True,
        shuffle=False)

    tgt_vocab = translators[0].fields['tgt'].vocab

    def var(a):
        return Variable(a, volatile=True)

    assert opt.beam_size == 1
    report_stats = onmt.Statistics()
    bos_id = tgt_vocab.stoi[onmt.IO.BOS_WORD]
    eos_id = tgt_vocab.stoi[onmt.IO.EOS_WORD]
    softmax = torch.nn.Softmax()

    for bid, batch in enumerate(test_iter):
        payloads = []
        _, src_lengths = batch.src
        context_lengths = src_lengths.repeat(opt.beam_size)
        for translator in translators:
            context, dec_states = translator.init_decoder_state(batch, test_data)
            payloads.append((dec_states, context))

        uniform_distrib = tt.FloatTensor([[1.] * opt.topk] * batch.batch_size)
        end_mask = tt.ByteTensor(batch.batch_size).fill_(0)

        # steps includes the <s> symbol
        n_steps = batch.tgt.size(0) if opt.explore_type == 'teacher_forcing' else opt.max_sent_length

        inp_tensor = tt.LongTensor(batch.batch_size).fill_(bos_id)
        inp = var(inp_tensor).view(1, -1)
        tgt = []
        selected_distrib = []
        selected_indices = []
        for step in range(1, n_steps):
            output = tt.FloatTensor(batch.batch_size, len(tgt_vocab.itos)).zero_()
            for j in range(len(translators)):
                dec_states, context = payloads[j]
                output_, dec_states = translators[j].step(inp, context, dec_states, context_lengths)
                output += output_.view(batch.batch_size, -1)
                payloads[j] = dec_states, context
            output.div_(len(translators))

            if opt.explore_type == 'teacher_forcing':
                output *= opt.distill_alpha
                ind = batch.tgt[step].data.view(-1, 1)
                temp_output = output.gather(1, ind)
                temp_output += (1 - opt.distill_alpha)
                output.scatter_(1, ind, temp_output)
            distrib, indices = torch.topk(output, opt.topk)
            if opt.annealing != 1:
                distrib.pow_(1. / opt.annealing)
            if opt.renormalize:
                distrib = softmax(var(torch.log(distrib))).data

            selected_distrib.append(distrib)
            selected_indices.append(indices)

            if opt.explore_type == 'teacher_forcing':
                inp_tensor = batch.tgt[step].data.view(1, -1)
            elif opt.explore_type == 'epsilon_greedy':
                if opt.gpu >= 0:
                    mask = (torch.rand(batch.batch_size, 1) > opt.epsilon_greedy_epsilon).long().cuda()
                else:
                    mask = (torch.rand(batch.batch_size, 1) > opt.epsilon_greedy_epsilon).long()
                ind = torch.distributions.Categorical(uniform_distrib).sample().view(-1, 1)
                inp_tensor = indices[:, 0].contiguous().view(-1, 1) * mask + indices.gather(1, ind) * (1 - mask)
                inp_tensor = inp_tensor.view(1, -1)
            else:
                ind = torch.distributions.Categorical(distrib).sample().view(-1, 1)
                inp_tensor = indices.gather(1, ind).view(1, -1)

            inp_tensor.masked_fill_(end_mask, eos_id)
            end_mask = (inp_tensor == eos_id)
            tgt.append(inp_tensor.view(-1))

            inp = var(inp_tensor)

            if end_mask.sum() == batch.batch_size:
                break
        if end_mask.sum() != batch.batch_size:
            # loop stop when not all batch ends and step reach the maximum
            inp_tensor.masked_fill_(tt.ByteTensor(batch.batch_size).fill_(1), eos_id)
            inp = var(inp_tensor)
            tgt.append(inp_tensor.view(-1))

            output = tt.FloatTensor(batch.batch_size, len(tgt_vocab.itos)).zero_()
            for j in range(len(translators)):
                dec_states, context = payloads[j]
                output_, dec_states = translators[j].step(inp, context, dec_states, context_lengths)
                output += output_.view(batch.batch_size, -1)
                payloads[j] = dec_states, context
            output.div_(len(translators))

            distrib, indices = torch.topk(output, opt.topk)
            if opt.renormalize:
                distrib = softmax(var(torch.log(distrib))).data

            selected_distrib.append(distrib)
            selected_indices.append(indices)

        tgt = torch.stack(tgt)
        selected_distrib = torch.stack(selected_distrib)
        selected_indices = torch.stack(selected_indices)
        n_steps = len(tgt)

        report_stats.update(get_batch_stats(batch, tgt, bos_id, eos_id))
        if (bid + 1) % opt.report_every == -1 % opt.report_every:
            report_stats.output(0, bid + 1, len(test_iter), report_stats.start_time)

        for b, i in enumerate(batch.indices.data.tolist()):
            effective_steps = list(itertools.takewhile(lambda s: tgt[s][b] != eos_id, range(n_steps)))
            test_data.examples[i].tgt = tuple([tgt_vocab.itos[tgt[step][b]] for step in effective_steps])

            effective_steps = effective_steps + [effective_steps[-1] + 1]
            test_data.examples[i].selected_distrib = [selected_distrib[step][b].tolist() for step in effective_steps]
            test_data.examples[i].selected_indices = [selected_indices[step][b].tolist() for step in effective_steps]

        if opt.verbose:
            for i in batch.indices.data.tolist():
                print(test_data.examples[i].tgt)
                print(test_data.examples[i].selected_distrib)
                print(test_data.examples[i].selected_indices)

    test_data.fields = []
    torch.save(test_data, open(opt.save_data + '.pt', 'wb'))

    # For selected_indices and selected_distrib, init_token is not needed.
    pad_info = {
        'selected_indices': {
            'eos_token': [fields['tgt'].vocab.stoi[fields['tgt'].eos_token]] * opt.topk,
            'pad_token': [fields['tgt'].vocab.stoi[fields['tgt'].pad_token]] * opt.topk,
        },
        'selected_distrib': {
            'eos_token': [0.] * opt.topk,
            'pad_token': [0.] * opt.topk,
        }
    }
    torch.save(pad_info, open(opt.save_pad + '.pt', 'wb'))
    print('Distillation data is generated.')


if __name__ == "__main__":
    generate()

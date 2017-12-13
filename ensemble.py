#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import argparse
import random
import codecs
import onmt
import onmt.IO
from onmt.Utils import use_gpu
import opts
import torch
from torch.autograd import Variable
import torchtext

parser = argparse.ArgumentParser(description='ensemble.py')
opts.add_md_help_argument(parser)
opts.ensemble_opts(parser)

opt = parser.parse_args()
random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.batch_size != 1:
    print("WARNING: -batch_size isn't supported currently, "
          "we set it to 1 for now!")
    opt.batch_size = 1


def stoi(sent, str2index):
    sent = sent + ' ' + onmt.IO.EOS_WORD
    return [str2index[word] for word in sent.split()]


def itos(indices, distrib, index2str):
    return map(list, zip(*[(index2str[i], p)
                           for i, p in zip(indices, distrib) if index2str[i] != onmt.IO.PAD_WORD]))


def ensemble():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translators = []
    for model in opt.models.split(','):
        # tricky, override the model path.
        opt.model = model
        translators.append(onmt.Translator(opt, dummy_opt.__dict__))
    n_translators = len(translators)
    assert n_translators > 0
    print('Created {0} translators.'.format(n_translators))

    test_data = torch.load(opt.data + '.pt')
    fields = onmt.IO.ONMTDataset.load_fields(torch.load(opt.vocab + '.pt'))
    fields = dict([(k, f) for (k, f) in fields.items() if k in test_data.examples[0].__dict__])

    test_data.fields = fields
    test_iter = onmt.IO.OrderedIterator(
        dataset=test_data, batch_size=opt.batch_size,
        device=opt.gpu if opt.gpu > -1 else -1,
        train=False, sort=False,
        shuffle=False)

    src_vocab = translators[0].fields['src'].vocab
    tgt_vocab = translators[0].fields['tgt'].vocab

    def var(a):
        return Variable(a, volatile=True)

    if opt.explore_type == 'translate':
        output_handler = codecs.open(opt.output, 'w', encoding='utf-8')
    else:
        output_handler = None

    assert opt.beam_size == 1
    for i, batch in enumerate(test_iter):

        payloads = []
        for translator in translators:
            context, dec_states = translator.init_decoder_state(batch, test_data)
            payloads.append((None, dec_states, context))

        input_ = var(torch.LongTensor([tgt_vocab.stoi[onmt.IO.BOS_WORD]])).view(1, 1, 1)
        if use_gpu(opt):
            input_.cuda()

        selected_distrib, selected_indices = [], []
        if opt.explore_type == 'teacher_forcing' or opt.explore_type == 'teach_force':
            n_steps = batch.tgt.size(0)
            for step in range(n_steps):
                for j in range(n_translators):
                    _, dec_states, context = payloads[j]
                    output, dec_states = translators[j].step(input_, context, dec_states)
                    payloads[j] = output, dec_states, context
                # print output
                output = sum([payload[0] for payload in payloads]) / len(translators)
                values, indices = torch.topk(output, opt.topk)

                selected_distrib.append(values.view(-1).tolist())
                selected_indices.append(indices.view(-1).tolist())

                input_ = batch.tgt[step][0].view(1, 1, 1)
                if use_gpu(opt):
                    input_.cuda()

            index = batch.indices.data.type(torch.IntTensor)[0]
            test_data.examples[index].selected_distrib = selected_distrib
            test_data.examples[index].selected_indices = selected_indices

        elif opt.explore_type == 'epsilon_greedy' or opt.explore_type == 'translate':
            tgt = []
            for step in range(opt.max_sent_length):
                for j in range(n_translators):
                    _, dec_states, context = payloads[j]
                    output, dec_states = translators[j].step(input_, context, dec_states)
                    payloads[j] = output, dec_states, context
                # print output
                output = sum([payload[0] for payload in payloads]) / len(translators)
                values_var, indices_var = torch.topk(output, opt.topk)
                values = values_var.view(-1).tolist()
                indices = indices_var.view(-1).tolist()

                selected_distrib.append(values)
                selected_indices.append(indices)

                if opt.explore_type == 'epsilon_greedy':
                    flip_coin = random.random()
                    if flip_coin < opt.epsilon_greedy_epsilon:
                        pred_id = random.choice(indices)
                    else:
                        pred_id = indices[0]
                else:
                    pred_id = indices[0]

                tgt.append(tgt_vocab.itos[pred_id])
                if tgt[-1] == onmt.IO.EOS_WORD:
                    break

                input_ = var(torch.LongTensor([pred_id])).view(1, 1, 1)
                if use_gpu(opt):
                    input_.cuda()

            if opt.explore_type == 'epsilon_greedy':
                index = batch.indices.data.type(torch.LongTensor)[0]
                test_data.examples[index].selected_distrib = selected_distrib
                test_data.examples[index].selected_indices = selected_indices
            else:
                print(' '.join(tgt[:-1]), file=output_handler)
        else:
            tgt = []
            for step in range(opt.max_sent_length):
                for j in range(n_translators):
                    _, dec_states, context = payloads[j]
                    output, dec_states = translators[j].step(input_, context, dec_states)
                    payloads[j] = output, dec_states, context
                # print output
                output = sum([payload[0] for payload in payloads]) / len(translators)
                values_var, indices_var = torch.topk(output, opt.topk)
                values = values_var.view(-1).tolist()
                indices = indices_var.view(-1).tolist()

                selected_distrib.append(values)
                selected_indices.append(indices)

                pred_id = random.choice(indices, p=values)
                tgt.append(tgt_vocab.itos[pred_id])
                if tgt[-1] == onmt.IO.EOS_WORD:
                    break

                input_ = var(torch.LongTensor([pred_id])).view(1, 1, 1)
                if use_gpu(opt):
                    input_.cuda()

            index = batch.indices.data.type(torch.LongTensor)[0]
            test_data.examples[index].selected_distrib = selected_distrib
            test_data.examples[index].selected_indices = selected_indices

    if opt.explore_type != 'translate':
        test_data.fields = []
        torch.save(test_data, open(opt.save_data + '.pt', 'wb'))

        pad_info = {
            'selected_indices': {
                'eos_token': [fields['tgt'].vocab.stoi[fields['tgt'].eos_token]] * opt.topk,
                'pad_token': [fields['tgt'].vocab.stoi[fields['tgt'].pad_token]] * opt.topk,
                'init_token': [fields['tgt'].vocab.stoi[fields['tgt'].init_token]] * opt.topk,
            },
            'selected_distrib': {
                'eos_token': [0.] * opt.topk,
                'pad_token': [0.] * opt.topk,
                'init_token': [0.] * opt.topk,
            }
        }
        torch.save(pad_info, open(opt.save_pad + '.pt', 'wb'))
        print('Distillation data is generated.')


if __name__ == "__main__":
    ensemble()

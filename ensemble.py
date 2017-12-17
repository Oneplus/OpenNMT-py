#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import argparse
import codecs
import onmt
import onmt.IO
import opts
import torch
from torch.autograd import Variable
import glob
import itertools

parser = argparse.ArgumentParser(description='ensemble.py')
opts.add_md_help_argument(parser)
opts.ensemble_opts(parser)

opt = parser.parse_args()


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


def ensemble():
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    tt = torch.cuda if opt.cuda else torch

    translators = load_translators()

    test_data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translators[0].fields, None)
    test_iter = onmt.IO.OrderedIterator(
        dataset=test_data, batch_size=opt.batch_size,
        device=opt.gpu,
        train=False, sort=False, shuffle=False)

    tgt_vocab = translators[0].fields['tgt'].vocab

    def var(a):
        return Variable(a, volatile=True)

    assert opt.beam_size == 1
    bos_id = tgt_vocab.stoi[onmt.IO.BOS_WORD]
    eos_id = tgt_vocab.stoi[onmt.IO.EOS_WORD]
    itos = tgt_vocab.itos

    output_handler = codecs.open(opt.output, 'w', encoding='utf-8')
    for bid, batch in enumerate(test_iter):
        payloads = []
        _, src_lengths = batch.src
        context_lengths = src_lengths.repeat(opt.beam_size)
        for translator in translators:
            context, dec_states = translator.init_decoder_state(batch, test_data)
            payloads.append((dec_states, context))

        n_translators_mask = tt.FloatTensor(batch.batch_size, len(tgt_vocab.itos)).fill_(len(translators))
        end_mask = tt.ByteTensor(batch.batch_size).fill_(0)
        n_steps = opt.max_sent_length

        inp_tensor = tt.LongTensor(batch.batch_size).fill_(bos_id)
        inp = var(inp_tensor).view(1, -1)
        tgt = []
        for step in range(1, n_steps):
            output = tt.FloatTensor(batch.batch_size, len(tgt_vocab.itos)).zero_()
            for j in range(len(translators)):
                dec_states, context = payloads[j]
                output_, dec_states = translators[j].step(inp, context, dec_states, context_lengths)
                output += output_.view(batch.batch_size, -1)
                payloads[j] = dec_states, context
            output = output.div(n_translators_mask)
            _, indices = torch.topk(output, 1)
            inp_tensor = indices[:, 0].contiguous().view(1, -1)
            inp_tensor.masked_fill_(end_mask, eos_id)
            end_mask = (inp_tensor == eos_id)
            tgt.append(inp_tensor.view(-1))

            inp = var(inp_tensor)

            if end_mask.sum() == batch.batch_size:
                break

        tgt = torch.stack(tgt)
        n_steps = len(tgt)

        print(bid)
        for b, i in enumerate(batch.indices.data.tolist()):
            effective_steps = list(itertools.takewhile(lambda s: tgt[s][b] != eos_id, range(n_steps)))
            print(' '.join(itos[tgt[step][b]] for step in effective_steps), file=output_handler)


if __name__ == "__main__":
    ensemble()

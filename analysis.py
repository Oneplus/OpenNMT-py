#!/usr/bin/env python
import argparse
import torch
import codecs


class Statistic(object):
    def __init__(self):
        self.p_at = {i: 0 for i in (1, 2, 5, 10)}
        self.p = 0
        self.n = 0
        self.p_mass = 0

    def update_p_mass(self, p_mass):
        self.p_mass += p_mass
        self.n += 1

    def update_p_at(self, indices, ind):
        if ind in indices[:1]:
            self.p_at[1] += 1
        if len(indices) >= 2 and ind in indices[:2]:
            self.p_at[2] += 1
        if len(indices) >= 5 and ind in indices[:5]:
            self.p_at[5] += 1
        if len(indices) >= 10 and ind in indices[:10]:
            self.p_at[10] += 1
        self.p += 1

    def report_p_mass(self):
        print('prob mass approximate: {0:.5f}'.format(self.p_mass / self.n))

    def report_p_at(self):
        print('P@1 {0:.5f}'.format(self.p_at[1] / self.p))
        print('P@2 {0:.5f}'.format(self.p_at[2] / self.p))
        print('P@5 {0:.5f}'.format(self.p_at[5] / self.p))
        print('P@10 {0:.5f}'.format(self.p_at[10] / self.p))


def load_tgt_vocab(filename):
    vocab = torch.load(filename)
    vocab = {k: v for k, v in vocab}
    tgt_vocab = vocab['tgt']
    return tgt_vocab


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('-data', required=True, help='the path to the .pt data file.')
    cmd.add_argument('-dump_tgt', default=False, action='store_true',
                     help='Output the target side string.')
    cmd.add_argument('-dump_tgt_details', default=False, action='store_true',
                     help='Detailedly output the target side.')
    cmd.add_argument('-dump_tgt_details_w_prob', default=False, action='store_true',
                     help='Detailedly output the target side.')
    cmd.add_argument('-eval_p_mass', default=False, action='store_true',
                     help='evaluate probability mass approximation.')
    cmd.add_argument('-eval_p_at', default=False, action='store_true',
                     help='evaluate P@1, require setting -src.')
    cmd.add_argument('-vocab', help='the path to the .pt vocab file.')
    cmd.add_argument('-src', help='the path to the source data.')

    args = cmd.parse_args()
    data = torch.load(args.data)

    if args.dump_tgt_details or args.eval_p_at:
        tgt_vocab = load_tgt_vocab(args.vocab)

    if args.eval_p_at:
        src = [line.strip().split() for line in codecs.open(args.src, 'r', encoding='utf-8')]
        assert len(data) == len(src)

    stat = Statistic()

    for i in range(len(data)):
        example = data.examples[i]
        if args.eval_p_mass:
            for distrib in example.selected_distrib:
                stat.update_p_mass(sum(distrib))

        if args.eval_p_at:
            source = src[i]
            for j in range(len(source)):
                if j >= len(example.selected_indices):
                    break
                indices = example.selected_indices[j]
                tgt_id = tgt_vocab.stoi[source[j]]
                stat.update_p_at(indices, tgt_id)

        if args.dump_tgt:
            print(' '.join(example.tgt))

        if args.dump_tgt_details:
            for j in range(len(example.selected_indices)):
                select_indices = example.selected_indices[j][:10]
                select_distrib = example.selected_distrib[j][:10]
                if args.dump_tgt_details_w_prob:
                    print(' ||| '.join(['{0}|{1:.5f}'.format(tgt_vocab.itos[ind], prob)
                                        for ind, prob in zip(select_indices, select_distrib)]))
                else:
                    print(' ||| '.join([tgt_vocab.itos[ind]
                                        for ind, prob in zip(select_indices, select_distrib)]))
            print()

    if args.eval_p_mass:
        stat.report_p_mass()

    if args.eval_p_at:
        stat.report_p_at()


if __name__ == "__main__":
    main()

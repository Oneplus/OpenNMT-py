#!/usr/bin/env python
import argparse
import torch
import codecs


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('-data', help='the path to the .pt data file.')
    cmd.add_argument('-vocab', help='the path to the .pt vocab file.')
    cmd.add_argument('-src', help='the path to the source data.')
    args = cmd.parse_args()
    data = torch.load(args.data)
    vocab = torch.load(args.vocab)
    vocab = {k: v for k, v in vocab}
    tgt_vocab = vocab['tgt']
    src = [line.strip().split() for line in codecs.open(args.src, 'r', encoding='utf-8')]
    assert len(data) == len(src)

    average_prob = 0
    n = 0
    p_at_1, p_at_2, p_at_5, p_at_10, p = 0, 0, 0, 0, 0
    for i in range(len(data)):
        example = data.examples[i]
        for distrib in example.selected_distrib:
            average_prob += sum(distrib)
            n += 1
        source = src[i]
        for j in range(len(source)):
            indices = example.selected_indices[j]
            tgt_id = tgt_vocab.stoi[source[j]]
            if tgt_id in indices[:1]:
                p_at_1 += 1
            if len(indices) >= 2 and tgt_id in indices[:2]:
                p_at_2 += 1
            if len(indices) >= 5 and tgt_id in indices[:5]:
                p_at_5 += 1
            if len(indices) >= 10 and tgt_id in indices[:10]:
                p_at_10 += 1
            p += 1
    average_prob /= n
    print('prob mass approximate: {0}'.format(average_prob))
    print('p@1 {0}'.format(p_at_1 / p))
    print('p@2 {0}'.format(p_at_2 / p))
    print('p@5 {0}'.format(p_at_5 / p))
    print('p@10 {0}'.format(p_at_10 / p))


if __name__ == "__main__":
    main()

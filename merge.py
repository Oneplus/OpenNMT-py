#!/usr/bin/env python
import argparse
import torch


def main():
    cmd = argparse.ArgumentParser('merge.py')
    cmd.add_argument('-reference', help='the path to the references')
    cmd.add_argument('-inputs', nargs='+', help='The paths ot the input files.')
    cmd.add_argument('-output', help='the path to the output data.')
    opt = cmd.parse_args()
    test_data = torch.load(opt.reference)

    reference = set()
    for ex in test_data.examples:
        reference.add(ex.indices, ex.tgt)

    for filename in opt.inputs:
        new_test_data = torch.load(filename)
        new_examples = [ex for ex in new_test_data.examples if (ex.indices, ex.tgt) not in reference]
        test_data.examples.extend(new_examples)

    indices = 0
    for ex in test_data.examples:
        ex.indices = indices
        indices += 1

    torch.save(test_data, open(opt.output, 'wb'))


if __name__ == "__main__":
    main()

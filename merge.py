#!/usr/bin/env python
import argparse
import torch


def main():
    cmd = argparse.ArgumentParser('merge.py')
    cmd.add_argument('-inputs', nargs='+', help='The paths ot the input files.')
    cmd.add_argument('-output', help='the path to the output data.')
    opt = cmd.parse_args()
    test_data = torch.load(opt.inputs[0])
    for filename in opt.inputs[1:]:
        new_test_data = torch.load(filename)
        test_data.examples.extend(new_test_data.examples)

    indices = 0
    for ex in test_data.examples:
        ex.indices = indices
        indices += 1

    torch.save(test_data, open(opt.output, 'wb'))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import torch


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('-data', help='the path to the .pt data file.')
    args = cmd.parse_args()
    data = torch.load(args.data)

    average_prob = 0
    n = 0
    for example in data.examples:
        for distrib in example.selected_distrib:
            average_prob += sum(distrib)
            n += 1
    average_prob /= n
    print(average_prob)


if __name__ == "__main__":
    main()

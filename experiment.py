#!/usr/bin/env python
import os
import argparse


def main():
    cmd = argparse.ArgumentParser('experiment.py')
    cmd.add_argument('-n', required=True, type=int, help='the number of models')
    cmd.add_argument('-explore_type', required=True, help='the exploration type [teacher_forcing, boltzmann]')
    cmd.add_argument('-alpha', default=1., help='the alpha, only used when explore_type=teacher forcing.')
    cmd.add_argument('-epsilon', default=0, help='the epsilon in the ')
    cmd.add_argument('-topk', required=True, type=int, help='the number of k-best')
    cmd.add_argument('-renormalize', default=False, action='store_true', help='do re-normalize.')
    opts = cmd.parse_args()
    explore_type = opts.explore_type
    assert explore_type in ('teacher_forcing', 'boltzmann', 'epsilon_greedy')
    if explore_type == 'teacher_forcing':
        explore_type += '_{0}'.format(opts.alpha)
    elif explore_type == 'epsilon_greedy':
        explore_type += '_{0}'.format(opts.epsilon)

    params = 'x{0}.{1}.top{2}.{3}'.format(opts.n, explore_type, opts.topk, 'norm' if opts.renormalize else 'nonorm')
    directory = os.path.join('saves', params)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if opts.n < 10:
        models = './model/lstm.h256.seed[1-{0}].pt'.format(opts.n)
    else:
        models = './model/lstm.h256.seed*.pt'

    with open(os.path.join(directory, 'generate.sh'), 'w') as handler:
        cmds = ['python', 'generate.py',
                '-vocab', 'iwslt2014/data.vocab',
                '-save_pad', '{dir}/data.pad'.format(dir=directory),
                '-models', '\'{models}\''.format(models=models),
                '-explore_type', '{explore}'.format(explore=opts.explore_type),
                '-report_every', '1',
                '-topk', '{k}'.format(k=opts.topk),
                '-gpu', '0']
        # Generate scripts
        if explore_type == 'teacher_forcing':
            cmds.extend(['-distill_alpha', '{alpha}'.format(alpha=opts.alpha)])
        elif explore_type == 'epsilon_greedy':
            cmds.extend(['-epsilon_greedy_epsilon', '{epsilon}'.format(epsilon=opts.epsilon)])

        if opts.renormalize:
            cmds = cmds + ['-renormalize']
        for fold in ('train', 'valid'):
            extra_cmds = ['-data', 'iwslt2014/data.{fold}'.format(fold=fold),
                          '-save_data', '{dir}/data.{fold}'.format(dir=directory, fold=fold)]
            print(' '.join(cmds + extra_cmds), file=handler)

    for optim in ('adam', 'sgd'):
        optim_cmds = ['-optim', optim]
        if optim == 'adam':
            optim_cmds += ['-learning_rate', '0.001']
        with open(os.path.join(directory, 'distill_{optim}.sh'.format(optim=optim)), 'w') as handler:
            cmds = ['nohup', 'python', 'train.py',
                    '-distill',
                    '-data', '{dir}/data'.format(dir=directory),
                    '-save_model', '{dir}/distill-model-{optim}'.format(dir=directory, optim=optim),
                    '-rnn_size', '256',
                    '-gpuid', '0',
                    '-seed', '1',
                    '-epochs', '30',
                    '-valid_txt', 'data/src-val.txt',
                    '-valid_output', '{dir}/val-runtime-output.txt'.format(dir=directory),
                    '-valid_script', './tools/iwslt_val_bleu.sh']
            log_cmds = ['>', '{dir}/distill_{optim}.log'.format(dir=directory, optim=optim), '&']
            print(' '.join(cmds + optim_cmds + log_cmds), file=handler)

        with open(os.path.join(directory, 'translate_{optim}.sh'.format(optim=optim)), 'w') as handler:
            cmds = ['python', 'translate.py',
                    '-gpu', '0',
                    '-model', '{dir}/distill-model-{optim}.pt'.format(dir=directory, optim=optim),
                    '-beam_size', '1',
                    '-max_sent_length', '50']
            for fold in ('test', 'val'):
                extra_cmds = [
                    '-src', 'data/src-{fold}.txt'.format(fold=fold),
                    '-tgt', 'data/tgt-{fold}.txt'.format(fold=fold),
                    '-output', '{dir}/{fold}-{optim}-output.txt'.format(dir=directory, optim=optim, fold=fold)]
                print(' '.join(cmds + extra_cmds), file=handler)


if __name__ == "__main__":
    main()

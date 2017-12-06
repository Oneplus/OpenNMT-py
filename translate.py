#coding:utf8

from __future__ import division
from builtins import bytes
import os
import sys
import argparse
import math
import codecs
import torch
import cPickle as pkl
from torch.autograd import  Variable


import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
if opt.batch_size != 1:
    print("WARNING: -batch_size isn't supported currently, "
          "we set it to 1 for now!")
    opt.batch_size = 1


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)
def sent_2_id(sent, str2index):
    sent = sent + ' ' + onmt.IO.EOS_WORD
    return [str2index[word] for word in sent.split()]

def id_2_sent(indices, distrib, index2str):
    return map(list, zip(*[(index2str[id],p) for id,p in zip(indices,distrib) 
                                if index2str[id] != onmt.IO.PAD_WORD]))

def generate_ensemble(mode='generate'):
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    opt.model = 'model/model_1.pt'
    translator_1 = onmt.Translator(opt, dummy_opt.__dict__)
    opt.model = 'model/model_2.pt'
    translator_2 = onmt.Translator(opt, dummy_opt.__dict__)
    opt.model = 'model/model_3.pt'
    translator_3 = onmt.Translator(opt, dummy_opt.__dict__)
    
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator_1.fields, None)
    
    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)
     
    src_vocab = translator_1.fields['src'].vocab
    tgt_vocab = translator_1.fields['tgt'].vocab
     
    def var(a): return Variable(a, volatile = True)

    if mode == 'generate':
        out_file = codecs.open(opt.output, 'w', 'utf-8')
    else:
        sentences = [line.strip() for line in open(opt.tgt)]
    
    distribution = []
     
    for i,batch in enumerate(test_data):
        sys.stdout.write(str(i * 100.0 / len(test_data)) + ' %\r')
        
        context_1, decStates_1 = translator_1.init_decoder_state(batch, data)
        context_2, decStates_2 = translator_2.init_decoder_state(batch, data)
        context_3, decStates_3 = translator_3.init_decoder_state(batch, data)
        
        input = var(torch.LongTensor([tgt_vocab.stoi[onmt.IO.BOS_WORD]])).view(1,1,1).cuda()
        
        distrib = []
        if mode == 'generate':
            pred_ids = []
            for i in range(opt.max_sent_length):
                # 预测到终止符
                output_1, decStates_1 = translator_1.step(input, context_1, decStates_1)
                output_2, decStates_2 = translator_2.step(input, context_2, decStates_2)
                output_3, decStates_3 = translator_3.step(input, context_3, decStates_3)
                #print output
                output = (output_1 + output_2 + output_3) / 3
                values, indices = torch.topk(output, 10)
                distrib.append([values.view(-1).tolist(),indices.view(-1).tolist()])
                pred_id = indices.view(-1).tolist()[0]
                pred_ids.append(pred_id)
                input = var(torch.LongTensor([pred_ids[-1]])).view(1,1,1).cuda()
                
                if pred_ids[-1] == tgt_vocab.stoi[onmt.IO.EOS_WORD]: 
                    break
            if len(pred_ids) > 1: 
                sent, distrib = id_2_sent(pred_ids,distrib,tgt_vocab.itos)
                out_file.write(' '.join(sent[:-1]) + '\n')
                distribution.append(distrib)
        else:
            sent = sent_2_id(sentences[i], tgt_vocab.stoi)
            for j in range(len(sent)):
                # 预测到终止符
                output_1, decStates_1 = translator_1.step(input, context_1, decStates_1)
                output_2, decStates_2 = translator_2.step(input, context_2, decStates_2)
                output_3, decStates_3 = translator_3.step(input, context_3, decStates_3)
                #print output
                output = (output_1 + output_2 + output_3) / 3
                values, indices = torch.topk(output, 10)
                distrib.append((values.view(-1).tolist(),indices.view(-1).tolist()))
                input = var(torch.LongTensor([sent[j]])).view(1,1,1).cuda()
            distribution.append(distrib)
        
        sys.stdout.flush()
    pkl.dump(distribution, open('data/' + mode + '_prob.pkl', 'w'))

def main():

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator.fields, None)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                words = get_src_words(
                    src_sent, translator.fields["src"].vocab.itos)

                os.write(1, bytes('\nSENT %d: %s\n' %
                                  (sent_number, words), 'UTF-8'))

                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                os.write(1, bytes('PRED %d: %s\n' %
                                  (sent_number, best_pred), 'UTF-8'))
                print("PRED SCORE: %.4f" % best_score)

                if opt.tgt:
                    tgt_sent = ' '.join(gold_sent)
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (sent_number, tgt_sent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % gold_score)

                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        os.write(1, bytes("[%.4f] %s\n" % (score, sent),
                                 'UTF-8'))

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
    #if opt.tgt:
    #    generate_ensemble('gold')
    #else:
    #    generate_ensemble('generate')

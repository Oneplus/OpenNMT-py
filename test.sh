#!/bin/bash
#python translate.py -model a.pt -src data/src-train.txt -output tgt-train-gen.txt -beam_size 1 -batch_size 1 -gpu 3
#python translate.py -model model/distill_gen_model.pt -src data/src-val.txt -output distill_gen_valid.txt -beam_size 1 -batch_size 1  -gpu 2
#python translate.py -model model/distill_gen_model.pt -src data/src-test.txt -output distill_gen_test.txt -beam_size 1 -batch_size 1  -gpu 2
#python translate.py -model model/distill_gold_model.pt -src data/src-val.txt -output distill_gold_valid.txt -beam_size 1 -batch_size 1  -gpu 2
python translate.py -model model/distill_gold_model.pt -src data/src-test.txt -output distill_gold_test.txt -beam_size 1 -batch_size 1  -gpu 2

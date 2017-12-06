#!/bin/bash
#python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train-gen.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/Distill -shuffle 0
#python train.py -data data/demo -save_model ./model/model_1 -gpuid 0 -epochs 100 -batch_size 2 -seed 1 
#python train.py -data data/demo -save_model ./model/model_2 -gpuid 0 -epochs 100 -batch_size 2 -seed 2 
#python train.py -data data/demo -save_model ./model/model_3 -gpuid 0 -epochs 100 -batch_size 2 -seed 3 
#python train.py -data data/Distill -save_model ./model/distill_gen_model -rnn_size 256 -epochs 50 -batch_size 16 -seed 1 -gpuid 2 -prob data/generate_prob.pkl -topK 3
python train.py -data data/IWSLT2014 -save_model ./model/distill_gold_model -rnn_size 256 -epochs 50 -batch_size 16 -seed 1 -gpuid 3 -prob data/gold_prob.pkl -topK 3

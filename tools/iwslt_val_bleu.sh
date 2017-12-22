#!/bin/bash

perl ./tools/multi-bleu.perl ./data/tgt-val.txt < $1 | awk '{print $3}' | awk -F ',' '{print $1}'

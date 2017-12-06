#!/bin/bash
python ./tools/wrap.py data/IWSLT2014/valid.de-en.de data/IWSLT2014/valid.de-en.en
./tools/wrap_xml.pl en src.sgm DemoSystem < $1 > hyp.sgm
perl ./tools/mteval-v11b.pl -s ./src.sgm -r ./ref.sgm -t ./hyp.sgm -c 
rm *.sgm

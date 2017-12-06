#!/usr/bin/env python
#coding:utf8
import sys 

def main(src,ref):
    src_list = [line.strip() for line in open(src)]
    ref_list = [line.strip() for line in open(ref)]

    assert len(src_list) == len(ref_list), 'sentences num is not same'

    with open('src.sgm','w') as src_f, open('ref.sgm','w') as ref_f:
        src_f.write('<srcset setid="test" srclang="de">\n')
        src_f.write('<doc sysid="ref" docid="1" genre="Other" origlang="de">\n')
        ref_f.write('<refset setid="test" srclang="de" trglang="en">\n')
        ref_f.write('<doc sysid="ref" docid="1" genre="Other" origlang="de">\n')
        for id, (src_sent, ref_sent) in enumerate(zip(src_list, ref_list)):
            src_f.write('<seg id="'+str(id+1)+'">' + src_sent + '</seg>\n')
            ref_f.write('<seg id="'+str(id+1)+'">' + ref_sent + '</seg>\n')
        src_f.write('</doc>\n</srcset>\n')
        ref_f.write('</doc>\n</refset>\n')

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])

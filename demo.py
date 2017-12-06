#!/usr/bin/env python
#coding:utf8
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-is_true',action='store_true')

args = parser.parse_args()

if args.is_true:
    print 'true'

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:16:53 2018

@author: avanetten
"""

from subprocess import Popen, PIPE, STDOUT

###############################################################################
def file_len(fname):
    '''Return length of file'''
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except:
        return 0
    
###############################################################################
def run_cmd(cmd):
    '''Write to stdout, etc, THIS SCREWS UP NOHUP.OUT!!!'''
    p = Popen(cmd, stdout = PIPE, stderr = STDOUT, shell = True)
    while True:
        line = p.stdout.readline()
        if not line: break
        print (line.replace('\n', ''))
    return

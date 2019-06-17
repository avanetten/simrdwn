#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:13:03 2018

@author: avanetten
"""

import os
# import shutil

root = '/raid/data/spacenet/off_nadir/RGB-8bit/'

for d in os.listdir(root):
    print("d:", d)
    d2 = os.path.join(root, d, 'Pan-Sharpen')
    print("d2:", d2)
    for f in os.listdir(d2):
        forig = os.path.join(d2, f)
        fnew = forig.split('.')[0] + '.jpg'
        # convert
        cmd = 'convert' + ' ' + forig + ' ' + fnew
        print("cmd:", cmd)
        os.system(cmd)
        # remove original
        os.system('rm' + ' ' + forig)

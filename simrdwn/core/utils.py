#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:16:53 2018

@author: avanetten
"""

import numpy as np
from subprocess import Popen, PIPE, STDOUT
from statsmodels.stats.weightstats import DescrStatsW


###############################################################################
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """

    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    # weighted mean of data (equivalent to np.average(array, weights=weights))
    mean = weighted_stats.mean
    # standard deviation with default degrees of freedom correction
    std = weighted_stats.std
    # variance with default degrees of freedom correction
    var = weighted_stats.var

    return (mean, std, var)


###############################################################################
def twinx_function(x, raw=False):
    V = 3./x
    if raw:
        return V
    else:
        return ["%.1f" % z for z in V]
    # return [z for z in V]


###############################################################################
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x,
                        [x < x0],
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


###############################################################################
def _file_len(fname):
    '''Return length of file'''
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except:
        return 0


###############################################################################
def _run_cmd(cmd):
    '''Write to stdout, etc,(incompatible with nohup)'''
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print(line.replace('\n', ''))
    return

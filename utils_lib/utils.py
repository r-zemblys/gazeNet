#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
@emai
"""
import json
from collections import OrderedDict
import numpy as np


def round_up_to_odd(f, min_val = 3):
    """Rounds input value up to nearest odd number.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f) // 2 * 2 + 1)
    w = min_val if w < min_val else w
    return w


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def BoxMuller_gaussian(u1,u2):
  z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
  z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
  return z1,z2

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

class Config(object):
    def __init__(self, param_file):
        self.param_file = param_file
        self.read_params()

    class bcolors(object):
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def read_params(self, current_params = None):
        with open(self.param_file, 'r') as f:
            self.params = json.load(f, object_pairs_hook=OrderedDict)
        if not(current_params is None) and not (current_params == self.params):
            print "TRAINING PARAMETERS CHANGED"
            for k, p in current_params.iteritems():
                if not(p == self.params[k]):
                    print self.bcolors.WARNING + \
                          "%s: %s --> %s" % (k, p, self.params[k]) + \
                          self.bcolors.ENDC
            return True
    def save_params(self, params=None):
        if not(params):
            params = self.params
        else:
            self.params = params

        with open(self.param_file, 'w') as f:
            json.dump(params, f, indent=4)

def human_format(num, suffixes=['', 'K', 'M', 'G', 'T', 'P']):
    m = sum([abs(num/1000.0**x) >= 1 for x in range(1, len(suffixes))])
    val = num/1000.**m
    return '%.3f%s' % (val, suffixes[m])
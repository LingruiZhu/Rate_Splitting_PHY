#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:20:15 2021

@author: Lingrui
"""

import numpy as np

def llr_to_prob_bpsk(llrs):
    prob_positive = np.exp(llrs) / (1 + np.exp(llrs))
    prob_negative = 1 / (1 + np.exp(llrs))
    return prob_positive, prob_negative




import commpy.channelcoding as cc

intlv = cc.RandInterlv(length=10, seed=0)
a = np.array([1, 2, 3 , 4, 5, 6, 7, 8, 9, 10])
b = intlv.interlv(a)
a_hat = intlv.deinterlv(b)

c = np.array([1, 2, 3 , 4, 5, 6, 7, 8, 9, 10, 11, 12])
d = np.reshape(c, [-1,3])
e = np.reshape(d,[1, -1])

list_a = list()
list_a.append(np.array([1, 1, 1]))
list_a.append(np.array([2, 2, 2]))
list_a.append(np.array([3, 3, 3]))
print(list_a[0])

matrix_a = np.matrix(list_a)
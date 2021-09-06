#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:56:25 2021

@author: Lingrui Zhu
"""

import numpy as np
import math
from scipy import special as sp

def Q_inverse(x):
    y = np.sqrt(2)*sp.erfinv(1-2*x)
    return y

# calculate data rate with finite block length
def private_rate_calculation(h, power_private, epsilon, block_length, power_noise):
    # calculate data rate for user 1
    SNR_private_1 = h[0]*power_private[0]/(power_noise + h[0]*power_private[1])
    r_shannon_1 = np.log2(1 + SNR_private_1)
    dispersion_1 = 1 - math.pow((1+SNR_private_1), -2)
    reduction_1 = np.log2(math.exp(1)) * np.sqrt(dispersion_1/block_length) * Q_inverse(epsilon)
    rate_private_1 = r_shannon_1 #- reduction_1
    
    # private rate for user 2 
    SNR_private_2 = h[1]*power_private[1]/(power_noise + h[1]*power_private[0])
    r_shannon_2 = np.log2(1 + SNR_private_2)
    dispersion_2 = 1 - math.pow((1+SNR_private_2), -2)
    reduction_2 = np.log2(math.exp(1)) * np.sqrt(dispersion_2/block_length) * Q_inverse(epsilon)
    rate_private_2 = r_shannon_2 #- reduction_2
    
    return [rate_private_1, rate_private_2]
      

h = [1, 1]
power_private = [10000, 10]
epsilon = 1e-5
block_length = 100
power_noise = 1
rp1, rp2 = private_rate_calculation(h, power_private, epsilon, block_length, power_noise)
    


    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:23:07 2021

@author: Lingrui
"""

import numpy as np
import math
import commpy.channelcoding as cc
import commpy.modulation as mod
import commpy.channels as ch
import commpy.utilities as utils

# at first write a tx end

# parameter setting
n_symbols = 10
modulation_order = 2
n_bits = 1000

# generte bits
message_bits = np.random.randint(0, 2, n_bits)

# channel coding
memory = np.array(2, ndmin=1)
g_matrix = np.array((0o5, 0o7), ndmin=2)
trellis = cc.Trellis(memory, g_matrix)
code_bits = cc.conv_encode(message_bits, trellis)

# interleaver
intlv_length = int(len(message_bits)/0.5)
# intleaver = cc.RandInterlv(length=intlv_length, seed=0)  # 
intlv_bits = intleaver.interlv(code_bits)


# modulation
qam_size = 4
qam = mod.QAMModem(qam_size)
symbols = qam.modulate(intlv_bits)

# channel
param = (complex(1), complex(0))
channel = ch.SISOFlatChannel(noise_std=1, fading_param=param)

# receiver
rx_symbols = channel.propagate(symbols)

# QAM demodulate
demod_symbols = qam.demodulate(rx_symbols, demod_type='soft', noise_var=1)

# deinterleave
deintlv_bits = intleaver.deinterlv(demod_symbols)

# vtb decoding
message_bits_hat = cc.viterbi_decode(deintlv_bits, trellis, decoding_type='soft')
decoded_bits = message_bits_hat[:len(message_bits)]


# calculate bit error rate
error_vector = (message_bits + decoded_bits) % 2
ber = np.sum(error_vector) / len(message_bits) 





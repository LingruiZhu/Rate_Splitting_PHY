#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:28:48 2021

@author: Lingrui Zhu
"""
import numpy as np

class channel():
    def __init__(self):
        self.channel_matrix = None
        self.snr_db = None
    
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def set_snr(self, snr):
        self.snr_db = snr
    
    def get_snr(self):
        return self.snr_db
    
    def get_channel_matrix(self):
        return self.channel_matrix
    
    def set_modu_type(self, modu_type):
        if modu_type == 'BPSK':
            self.is_complex = False
        else:
            self.is_complex = True
    
    def propogate(self, tx_signal):
        channel_output = np.matmul(self.channel_matrix, tx_signal)
        snr_lin = 10 ** (self.snr_db/10)
        if self.is_complex:
            noise = 1/np.sqrt(2)*(np.random.normal(0, 1/snr_lin, [len(channel_output), 1]) + 
                              1j*np.random.normal(0, 1/snr_lin, [len(channel_output), 1]))
        else:
            noise = np.random.normal(0, 1/snr_lin, [len(channel_output), 1])
        received_signal = channel_output+noise
        return received_signal
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:33:32 2021

Rate splitting transmitter

@author: Lingrui Zhu
"""
import numpy as np
import commpy as cp
import commpy.channelcoding as cc
import commpy.channels as ch
import commpy.modulation as modem


class System_structure():
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.trellis = None
        self.interleaver = None
        self.modulator = None
    
    def set_trellis(self, trellis):
        self.trellis = trellis
        
    def set_intleaver(self, interleaver):
        self.interleaver = interleaver
    
    def set_modem(self, modulator):
        self.modulator = modulator


class Tx_process():
    def __init__(self, structs):
        """
        initialize the rate splitting transmitter and define the following attributes:
            n_t:            number of antennas
            n_streams:      number of streams to transmit
        """
        self.n_bits = structs.n_bits             # default settings
        
        self.coding_scheme = 'cc'       # now just one coding scheme: convolutional codes
        if self.coding_scheme == 'cc':
            self.trellis = structs.trellis
        self.interleaver = structs.interleaver
        self.modulator = structs.modulator
    
    
    def generate_data(self):
        self.msg_bits = np.random.randint(0, 2, size=self.n_bits)
    
    def channel_coding(self):
        self.code_bits = cc.conv_encode(self.msg_bits, self.trellis)
        
    def interleave(self):
        self.intleaved_code_bits = self.interleaver.interlv(self.code_bits)
    
    def modulate(self):
        self.symbols = self.modulator.modulate(self.intleaved_code_bits)
    
    def run(self):
        self.generate_data()
        self.channel_coding()
        self.interleave()
        self.modulate()
        return self.symbols
    
    def get_msg_bits(self):
        return self.msg_bits
    
class Transmitter_rate_splitting():
    def __init__(self, common_struct, private_struct_list, n_streams):
        self.common_struct = common_struct
        self.n_streams = n_streams
        self.common_chain = Tx_process(common_struct)
        self.tx_chain_list = list()
        self.msg_bits_list = list()
        self.tx_symbols_list = list()
        
        # add the common message at first
        common_tx_chain = Tx_process(common_struct)
        common_tx_symbols = common_tx_chain.run()
        common_msg_bits = common_tx_chain.get_msg_bits()
        
        self.tx_chain_list.append(common_tx_chain)
        self.msg_bits_list.append(common_msg_bits)
        self.tx_symbols_list.append(common_tx_symbols)
        
        # here assume that the parameters for private streams are the same
        for i in range(n_streams):
            tx_chain = Tx_process(private_struct_list[i])
            tx_symbols = tx_chain.run()
            msg_bits = tx_chain.get_msg_bits()
            
            self.tx_chain_list.append(tx_chain)
            self.msg_bits_list.append(msg_bits)
            self.tx_symbols_list.append(tx_symbols)
            
    def precoding(self, scheme, channel_matrix, noise_std):
        noise_var = np.power(noise_std, 2)
        n_r = channel_matrix.shape[0]
        n_t = channel_matrix.shape[1]
        
        # precoding for private part
        H = np.matrix(channel_matrix)
        w_mmse_temp = H.H * (H*H.H + noise_var*np.eye(n_r)).I       
        beta = np.sqrt(n_t/np.trace(w_mmse_temp))
        w_private_mmse = beta*w_mmse_temp
        
        w_common_ambf = np.transpose(np.sum(H, axis=0) / np.linalg.norm(np.sum(H, axis=0)))
        
        
        tx_symbols_matrix = np.matrix(self.tx_symbols_list)
        self.tx_signal = w_common_ambf*tx_symbols_matrix[0,:] + w_private_mmse*tx_symbols_matrix[1:,:]
        
    
    def get_msg_bits(self):
        return self.msg_bits_list
    
    def get_tx_symbols(self):
        return self.tx_symbols_list
    
    def get_tx_siganls(self):
        return self.tx_signal
    
    
class Receiver_rate_splitting():
    def __init__(self,):
        
    def signal_detection(self):
        
    def calculate_llr(self):
        
    def deinterleave(self):
    
    def decode(self):
        
    def hard_decision(self):
    
    def private_detect(self):
        
    def common_detect(self):

def main():
    n_bits = 10000
    
    memory = np.array(2, ndmin=1)
    g_matrix = np.array((0o5, 0o7), ndmin=2)
    trellis = cc.Trellis(memory, g_matrix)
    code_rate = trellis.k / trellis.n
    
    interleave_length = int(n_bits / code_rate)
    interleaver = cc.RandInterlv(length=interleave_length, seed=0)
    
    qam_size = 4
    qam_modulator = modem.QAMModem(qam_size)

    struct1 = System_structure(n_bits)
    struct1.set_trellis(trellis)
    struct1.set_intleaver(interleaver)
    struct1.set_modem(qam_modulator)
    
    struct2 = System_structure(n_bits)
    struct2.set_trellis(trellis)
    struct2.set_intleaver(interleaver)
    struct2.set_modem(qam_modulator)
    
    private_struct_list = list()
    private_struct_list.append(struct1)
    private_struct_list.append(struct2)
    
    structc = System_structure(n_bits)
    structc.set_trellis(trellis)
    structc.set_intleaver(interleaver)
    structc.set_modem(qam_modulator)
    
    transmitter_rs = Transmitter_rate_splitting(structc, private_struct_list, 2)
    msg_bits_list = transmitter_rs.get_msg_bits()
    tx_symbols = transmitter_rs.get_tx_symbols()

if __name__ == '__main__':
    # main()
    n_bits = 10000
    n_t = 4
    n_user = 2
    n_r = 1
    
    memory = np.array(2, ndmin=1)
    g_matrix = np.array((0o5, 0o7), ndmin=2)
    trellis = cc.Trellis(memory, g_matrix)
    code_rate = trellis.k / trellis.n
    
    interleave_length = int(n_bits / code_rate)
    interleaver = cc.RandInterlv(length=interleave_length, seed=0)
    
    qam_size = 4
    qam_modulator = modem.QAMModem(qam_size)

    struct1 = System_structure(n_bits)
    struct1.set_trellis(trellis)
    struct1.set_intleaver(interleaver)
    struct1.set_modem(qam_modulator)
    
    struct2 = System_structure(n_bits)
    struct2.set_trellis(trellis)
    struct2.set_intleaver(interleaver)
    struct2.set_modem(qam_modulator)
    
    private_struct_list = list()
    private_struct_list.append(struct1)
    private_struct_list.append(struct2)
    
    structc = System_structure(n_bits)
    structc.set_trellis(trellis)
    structc.set_intleaver(interleaver)
    structc.set_modem(qam_modulator)
    
    transmitter_rs = Transmitter_rate_splitting(structc, private_struct_list, 2)
    msg_bits_list = transmitter_rs.get_msg_bits()
    tx_symbols = transmitter_rs.get_tx_symbols()
    
    H = np.zeros((n_user*n_r, n_t), dtype=complex)
    for i in range(n_user*n_r):
        H[i,:] = np.random.randn(n_t) + np.random.randn(n_t)*(1j)
    
    noise_std = 0.5
    transmitter_rs.precoding(H, noise_std)
    tx_signal = transmitter_rs.get_tx_siganls()

    
        
        

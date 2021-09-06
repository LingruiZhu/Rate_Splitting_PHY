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
from scipy import stats

from utils import test


def llr_to_prob_bpsk(llrs):
    prob_positive = np.exp(llrs) / (1 + np.exp(llrs))
    prob_negative = 1 / (1 + np.exp(llrs))
    return prob_positive, prob_negative

def add_awgn_noise(input_signal, snr_db, rate):
    n_antenas = input_signal.shape[0]
    output_signal = np.zeros(input_signal.shape)
    for i in range(n_antenas):
        input_temp = input_signal[i,:]
        input_signal_squeezed = np.squeeze(np.asarray(input_temp))
        output_signal[i,:] = ch.awgn(input_signal_squeezed, snr_db, rate)
    
    return output_signal

class Channel():
    def __init__(self, channel_matrix):
        self.h_matrix = channel_matrix
    
    def propogate(self, input_signal):
        h = self.h_matrix
        input_array = np.asarray(input_signal)
        output_signal = np.matmul(h, input_array)
        return output_signal

class System_structure():
    def __init__(self, n_bits, is_bpsk):
        self.n_bits = n_bits
        self.trellis = None
        self.interleaver = None
        self.modulator = None
        self.is_bpsk = is_bpsk
        self.is_spc = False
    
    def set_trellis(self, trellis):
        self.trellis = trellis
    
    def set_spc(self, spc_k):
        self.is_spc = True
        self.spc_k = spc_k
    
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
        self.is_bpsk = structs.is_bpsk
        self.is_spc = structs.is_spc
        if structs.is_spc:
            self.spc_k = structs.spc_k
    
    def generate_data(self):
        self.msg_bits = np.random.randint(0, 2, size=self.n_bits)
    
    def channel_coding(self):
        self.code_bits = cc.conv_encode(self.msg_bits, self.trellis)
    
    def spc_coding(self):
        k = self.spc_k
        msg_bits = self.msg_bits
        info_bits = np.reshape(msg_bits, (-1,k))
        m, n = info_bits.shape
        
        code_bits = (-1) * np.ones([m, n+1])
        code_bits[:,:k] = info_bits
        code_bits[:,k] = np.sum(info_bits, axis=1)%2
        code_bits = code_bits.flatten()
        self.code_bits = code_bits
        
    def interleave(self, code_bits):            # TODO: turn on the interleave function again
        # self.intleaved_code_bits = self.interleaver.interlv(code_bits)
        self.intleaved_code_bits = code_bits
        return self.intleaved_code_bits
        
    
    def modulate(self):
        if self.is_bpsk:
            self.symbols = (-1) ** self.intleaved_code_bits
        else: 
            self.symbols = self.modulator.modulate(self.intleaved_code_bits)
    
    def run(self):
        self.generate_data()
        if self.is_spc:
            self.spc_coding()
        else:
            self.channel_coding()
        self.interleave(self.code_bits)
        self.modulate()
        return self.symbols
    
    def get_msg_bits(self):
        return self.msg_bits
    
    
class Transmitter_rate_splitting():
    def __init__(self, n_streams):
        self.n_streams = n_streams
        self.tx_chain_list = list()
        self.msg_bits_list = list()
        self.tx_symbols_list = list()
    
    def set_common_tx_process(self, common_tx):    
        # add the common message at first
        common_tx_symbols = common_tx.run()
        common_msg_bits = common_tx.get_msg_bits()
        
        self.tx_chain_list.append(common_tx)
        self.msg_bits_list.append(common_msg_bits)
        self.tx_symbols_list.append(common_tx_symbols)
    
    def set_private_tx_process(self, private_tx):
        # here assume that the parameters for private streams are the same
        tx_chain = private_tx
        tx_symbols = tx_chain.run()
        msg_bits = tx_chain.get_msg_bits()
            
        self.tx_chain_list.append(tx_chain)
        self.msg_bits_list.append(msg_bits)
        self.tx_symbols_list.append(tx_symbols)
        bits_list = self.msg_bits_list
        symbols_list = self.tx_symbols_list
        check_here = 1
        
            
    def precoding(self, channel_matrix, noise_std):
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
    
    def no_precoding(self):
        tx_symbols = self.tx_symbols_list
        tx_signal = np.matrix(self.tx_symbols_list)
        self.tx_signal = tx_signal
        return tx_signal
    
    def get_msg_bits(self):
        return self.msg_bits_list
    
    def get_tx_symbols(self):
        return self.tx_symbols_list
    
    def get_tx_siganls(self):
        return self.tx_signal
    
    
class Rx_process_rate_splitting():
    def __init__(self, symbol_length, code_rate):        
        self.received_signal = 0
        self.extrinsic_info = 0
        self.apriori_info = 0  # not sure if its necessary to set a attribute here.????
        self.interleave = None
        
        self.symbol_lenghth = symbol_length
        self.codeword_length = symbol_length
        self.infobits_length = int(self.codeword_length * code_rate)
        self.code_rate = code_rate
        
    def set_channel(self, channel_effective, channel_effective_dash):
        self.channel_effective = channel_effective
        self.channel_effective_dash = channel_effective_dash
        self.channel_dim0 = channel_effective.shape[0]
        self.channel_dim1 = channel_effective.shape[1]
        self.channel = Channel(channel_effective)
        
    def reconstruct(self, tx_process, apriori_info, channel):
        prob_pos, prob_neg = llr_to_prob_bpsk(apriori_info)
        bpsk_signal = 1*prob_pos + (-1)*prob_neg
        interleaved_signal = tx_process.interleave(bpsk_signal)
        reconstructed_signal = channel.propogate(np.expand_dims(interleaved_signal, axis=0))
        return reconstructed_signal
    
    def signal_detect(self, received_signal, tx_process, apriori_info, variance_noise):
        reconstructed_signal = self.reconstruct(tx_process, apriori_info, self.channel)
        received_signal_tilde = received_signal - reconstructed_signal
        
        # mmse equalizer
        channel_effective = np.matrix(self.channel_effective)
        channel_effective_dash = np.matrix(self.channel_effective_dash)
        E_x = 1;        # here to check the energy of bpsk signal
        dim0 = channel_effective.shape[0]
        dim1 = channel_effective.shape[1]
        equalizer_mmse = (channel_effective * (E_x*np.eye(dim1)) * channel_effective.H + 
                          channel_effective_dash * (E_x*np.eye(dim1)) * channel_effective_dash.H + 
                          variance_noise * np.eye(dim0)).I * (E_x * channel_effective)
        self.equalizer = equalizer_mmse
        estimated_signal = equalizer_mmse.H * received_signal_tilde
        return estimated_signal
        
    def calculate_llr(self, estimated_signal, apripri_info, extrinic_info, variance_noise):
        # calculate the likelihood ration symbol-wise (for BPSK bit-wise)
        # extrinsic information is not needed when BPSK adopted

        dim1 = self.channel_dim1
        dim0 = self.channel_dim0
        E_x = 1;       
        
        channel_effective = np.matrix(self.channel_effective)
        channel_effective_dash = np.matrix(self.channel_effective_dash)
        
        eq = self.equalizer
        chef = self.channel_effective
        mean_positive = self.equalizer.H * self.channel_effective*(+1)
        mean_negative = self.equalizer.H * self.channel_effective*(-1)
        variance = self.equalizer.H * (channel_effective * (E_x*np.eye(dim1)) * channel_effective.H + 
                   channel_effective_dash * (E_x*np.eye(dim1)) * channel_effective_dash.H + 
                   variance_noise * np.eye(dim0)) * self.equalizer
        variance = np.real(variance)                             
        num_real = stats.norm.pdf(np.real(estimated_signal), np.real(mean_positive), variance)
        num_imag = stats.norm.pdf(np.imag(estimated_signal), np.imag(mean_positive), variance)
        denum_real = stats.norm.pdf(np.real(estimated_signal), np.real(mean_negative), variance)
        denum_imag = stats.norm.pdf(np.imag(estimated_signal), np.imag(mean_negative), variance)
        probs = (num_real*num_imag) / (denum_real*denum_imag)
        llr = np.log(probs)
        # llr = stats.norm.pdf(np.real(estimated_signal), mean_positive, variance) / stats.norm.pdf(np.real(estimated_signal), mean_negative, variance)
        return llr
    
    def app_spc_decoder(self, soft_bits):
        # sofr decode (3,2) spc code
        soft_bits = np.reshape(soft_bits, [-1, 3])
        codeword_number = soft_bits.shape[0]
        codeword_length = soft_bits.shape[1]
        infobits_length = 2
        extrinsic_info = np.zeros([codeword_number, codeword_length])
        decoded_info_bits = np.zeros([codeword_number, infobits_length])
        
        for i in range(codeword_number):
            for j in range(codeword_length):
                bit_index = list(range(codeword_length))
                bit_index.remove(j)
                other_bits = soft_bits[i, bit_index]
                sign = np.prod(np.sign(other_bits))
                value = np.max(np.abs(other_bits))
                extrinsic_info[i,j] = sign*value
        
        llr = soft_bits + extrinsic_info
        info_bits_llr = llr[:,:2]
        decoded_info_bits = np.where(info_bits_llr>0, 0, 1)
        
        llr_flat = np.reshape(llr, [1, -1])
        decoded_info_bits_flat = np.reshape(decoded_info_bits, [1, -1])
        return llr_flat, decoded_info_bits_flat
    
    def deinterleave(self, input):
        deinterleaved = self.interleave.deinterleave(input)
        return deinterleaved
    
    
class Receiver_rate_splitting():
    def __init__(self):
        self.common_receiver = None
        self.private_receiver = None
        
    def set_common_receiver(self, common_receiver):
        self.common_receiver = common_receiver
        self.codeword_length = common_receiver.codeword_length
        self.infobits_length = common_receiver.infobits_length
        
    def set_private_receiver(self, private_receiver):
        self.private_receiver = private_receiver
    
    def set_common_tx_process(self, common_tx):
        self.common_tx_process = common_tx
        
    def set_private_tx_process(self, private_tx):
        self.private_tx_process = private_tx
    
    def set_snr(self, snr, signal_power):
        self.snr_db = snr
        self.snr_linear = 10 ** (snr/10)
        self.signal_power = signal_power
        self.noise_power = signal_power / self.snr_linear
        self.noise_variance = self.noise_power/2
    
    def decode(self, received_signal, iteration_number):
        decoded_common_bits = np.zeros([iteration_number, self.infobits_length])
        apriori_common_llrs = np.zeros([iteration_number, self.codeword_length])
        
        decoded_private_bits = np.zeros([iteration_number, self.infobits_length])
        apriori_private_llrs = np.zeros([iteration_number, self.codeword_length])
        for i in range(iteration_number):    # TODO: add the deinterleave part
            # at first iteration: common part
            index = np.max(i-1, 0)
            a = 1
            estimated_common_signal = self.common_receiver.signal_detect(received_signal, self.common_tx_process, apriori_private_llrs[index, :], 
                                                                         variance_noise = self.noise_power/2 )
            llr_common = self.common_receiver.calculate_llr(estimated_common_signal, apriori_private_llrs[index, :], extrinic_info=0,
                                                                         variance_noise = self.noise_variance)
            apriori_llrs_common_temp, decoded_bits_common_temp = self.common_receiver.app_spc_decoder(llr_common)
            apriori_common_llrs[i,:] = apriori_llrs_common_temp
            decoded_common_bits[i,:] = decoded_bits_common_temp
            
            # private part
            estimated_private_signal = self.private_receiver.signal_detect(received_signal, self.private_tx_process, apriori_common_llrs[i,:], variance_noise = self.noise_power/2)
            llr_private = self.private_receiver.calculate_llr(estimated_private_signal, apriori_common_llrs[i,:], extrinic_info=0,
                                                                         variance_noise = self.noise_variance)
            apriori_llrs_private_temp, decoded_bits_private_temp = self.private_receiver.app_spc_decoder(llr_private)
            apriori_private_llrs[i,:] = apriori_llrs_private_temp
            decoded_private_bits[i,:] = decoded_bits_private_temp
        
        return decoded_common_bits, decoded_private_bits
#        
#    def deinterleave(self):
#
#    
#    def private_detect(self):
#        
#    def common_detect(self):

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

    struct1 = System_structure(n_bits, is_bpsk=True)
    struct1.set_trellis(trellis)
    struct1.set_intleaver(interleaver)
    struct1.set_modem(qam_modulator)
    
    struct2 = System_structure(n_bits, is_bpsk=True)
    struct2.set_trellis(trellis)
    struct2.set_intleaver(interleaver)
    struct2.set_modem(qam_modulator)
    
    private_struct_list = list()
    private_struct_list.append(struct1)
    private_struct_list.append(struct2)
    
    structc = System_structure(n_bits, is_bpsk=True)
    structc.set_trellis(trellis)
    structc.set_intleaver(interleaver)
    structc.set_modem(qam_modulator)
    
    transmitter_rs = Transmitter_rate_splitting(structc, private_struct_list, 2)
    msg_bits_list = transmitter_rs.get_msg_bits()
    tx_symbols = transmitter_rs.get_tx_symbols()

if __name__ == '__main__':
    # main()
    n_bits = 10
    n_t = 3
    n_user = 2
    n_r = 2
    
    memory = np.array(2, ndmin=1)
    g_matrix = np.array((0o5, 0o7), ndmin=2)
    trellis = cc.Trellis(memory, g_matrix)
    code_rate = trellis.k / trellis.n
    
    code_rate = 2/3
    
    interleave_length = int(n_bits / code_rate)
    interleaver = cc.RandInterlv(length=interleave_length, seed=0)
    
    qam_size = 4
    qam_modulator = modem.QAMModem(qam_size)
    is_bpsk = True
    spc_k = 2

    struct1 = System_structure(n_bits, is_bpsk)
    struct1.set_trellis(trellis)
    struct1.set_intleaver(interleaver)
    struct1.set_modem(qam_modulator)
    struct1.set_spc(spc_k)
    
    struct2 = System_structure(n_bits, is_bpsk)
    struct2.set_trellis(trellis)
    struct2.set_intleaver(interleaver)
    struct2.set_modem(qam_modulator)
    struct2.set_spc(spc_k)
    
    private_struct_list = list()
    private_struct_list.append(struct1)
    private_struct_list.append(struct2)
    
    structc = System_structure(n_bits, is_bpsk)
    structc.set_trellis(trellis)
    structc.set_intleaver(interleaver)
    structc.set_modem(qam_modulator)
    structc.set_spc(spc_k)
    
    tx_process_common = Tx_process(structc)
    tx_process_private_1 = Tx_process(struct1)
    tx_process_private_2 = Tx_process(struct2)
    
    transmitter_rs = Transmitter_rate_splitting(n_streams = 2)
    transmitter_rs.set_common_tx_process(tx_process_common)
    transmitter_rs.set_private_tx_process(tx_process_private_1)
    transmitter_rs.set_private_tx_process(tx_process_private_2)
    
    msg_bits_list = transmitter_rs.get_msg_bits()
    tx_symbols = transmitter_rs.get_tx_symbols()
    
    H = np.zeros((n_user*n_r, n_t), dtype=complex)
    for i in range(n_user*n_r):
        H[i,:] = np.random.randn(n_t) + np.random.randn(n_t)*(1j)
    
    H1 = H[:2,:]
    H2 = H[2:,:]
    channel_1 = Channel(H1)
    channel_2 = Channel(H2)
    
    transmitter_rs.no_precoding()
    tx_signal = transmitter_rs.get_tx_siganls()
    
    # channel
    channel_output_1 = channel_1.propogate(tx_signal)
    channel_output_2 = channel_2.propogate(tx_signal)
    
    # MARK: try to only transmit common message and private message no.1 at first, then to debug
    tx_signal.
    
    # add noise
    rx1 = add_awgn_noise(channel_output_1, snr_db=10, rate=1)
    rx2 = add_awgn_noise(channel_output_2, snr_db=10, rate=1)
    
    # initiallize the receiver for user 1
    channel_common1 = np.expand_dims(H1[:,0], axis=1)
    channel_private1 = np.expand_dims(H1[:,1], axis=1)
    
    rx_process_common_1 = Rx_process_rate_splitting(symbol_length=rx1.shape[1], code_rate=2/3)
    rx_process_common_1.set_channel(channel_common1, channel_private1)
    rx_process_private_1 = Rx_process_rate_splitting(symbol_length=rx1.shape[1], code_rate=2/3)
    rx_process_private_1.set_channel(channel_private1, channel_common1)
    receiver_rs_1 = Receiver_rate_splitting()
    
    receiver_rs_1.set_common_receiver(rx_process_common_1)
    receiver_rs_1.set_private_receiver(rx_process_private_1)
    receiver_rs_1.set_common_tx_process(tx_process_common)
    receiver_rs_1.set_private_tx_process(tx_process_private_1)
    receiver_rs_1.set_snr(3, signal_power=1)
    common_bits_1, private_bits_1 = receiver_rs_1.decode(rx1, iteration_number=20)
    
    common_bits_1_it3 = common_bits_1[2]
    private_bits_1_it3 = private_bits_1[2]
    common_bits = msg_bits_list[0]
    private_1 = msg_bits_list[1]
    
    common_ber = np.sum(np.abs(common_bits-common_bits_1_it3)) / len(common_bits)
    print(common_bits_1_it3)
    print(common_bits_1)
    
    
    
        
        

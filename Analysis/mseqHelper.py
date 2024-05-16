#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:44:35 2021

@author: ravinderjit
"""

import numpy as np

def mseqXcorr(epochs,mseq):
    #epochs in shape channels x trials x time
    resp_m = epochs.mean(axis=1)
    resp_m = resp_m - resp_m.mean(axis=1)[:,np.newaxis]
    Ht_m = np.zeros([resp_m.shape[0],resp_m.shape[1]+mseq.size-1])
    
    mseq = mseq / np.sqrt(np.sum(mseq**2)) #normalize mseq by sqrt of total power
    
    for ch in range(resp_m.shape[0]):
        Ht_m[ch,:] = np.correlate(resp_m[ch,:],mseq,mode='full')#[mseq[m].size-1:]
    
    return Ht_m

def mseqXcorrEpochs(epochs,mseq):
    
    Ht_epochs = np.zeros([epochs.shape[0], epochs.shape[1], epochs.shape[2] + mseq.size-1])
    
    mseq = mseq / np.sqrt(np.sum(mseq**2)) #normalize mseq by sqrt of total power
    for ep in range(epochs.shape[1]):
        resp_m = epochs[:,ep,:]
        resp_m = resp_m - resp_m.mean(axis=1)[:,np.newaxis]
        Ht_ep = np.zeros([resp_m.shape[0],resp_m.shape[1]+mseq.size-1])
        for ch in range(resp_m.shape[0]):
            Ht_ep[ch,:] = np.correlate(resp_m[ch,:],mseq,mode='full')
            
        Ht_epochs[:,ep,:] = Ht_ep
        
    return Ht_epochs

 
def mseqXcorrEpochs_fft(epochs,mseq,fs):
    nfft = int(2**np.ceil(np.log2(mseq.size)))
    if mseq.size < fs:
        t_keep = np.arange(-mseq.size/fs,mseq.size/fs,1/fs)
    else:       
        t_keep = np.arange(-1,1,1/fs)
        
    half_keep = int(np.round(t_keep.size / 2))
    
    Ht_epochs = np.zeros([epochs.shape[0],epochs.shape[1],t_keep.size])
    
    mseq = mseq / np.sqrt(np.sum(mseq**2)) #normalize mseq by sqrt of total power
    mseq_f = np.fft.fft(mseq,n=nfft)
    mseq_f = mseq_f[np.newaxis,:]
    for ep in range(epochs.shape[1]):
        epoch_fft = np.fft.fft(epochs[:,ep,:] - np.mean(epochs[:,ep,:]),n=nfft)
        Ht_ep = np.fft.ifft(epoch_fft * np.conj(mseq_f))
        Ht_ep = np.concatenate([Ht_ep[:,-half_keep:],Ht_ep[:,:half_keep]],axis=1)
        Ht_epochs[:,ep,:] = Ht_ep
        
    return Ht_epochs, t_keep
        
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:00:53 2021

@author: ravinderjit
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr
from mseqHelper import mseqXcorrEpochs_fft


# from anlffr.spectral import mtspecraw
# from anlffr.spectral import mtplv
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
mseq_loc = 'mseqEEG_150_bits10_4096.mat'


file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + mseq_loc
Mseq_dat = sio.loadmat(file_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
mseq_shift = np.roll(mseq,int(np.round(mseq.size/2)))

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250']

for sub in range(len(Subjects)):
    subject = Subjects[sub]

    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=1,h_freq=200)
    
    if subject == 'S207':
        data_eeg.info['bads'].append('A15')
        
    if subject == 'S228':
        data_eeg.info['bads'].append('A25')
        data_eeg.info['bads'].append('A28')
        

#%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
    ocular_projs = Projs
    
    if subject == 'S211':
        ocular_projs = [Projs[0]] 
    
    if subject == 'S207':
        ocular_projs = [Projs[0]]
    
    if subject == 'S228':
        ocular_projs = [Projs[0]]
        
    if subject == 'S236':
        ocular_projs = [Projs[0]] 
        
    if subject == 'S238':
         ocular_projs = [Projs[0]] 
         
    if subject == 'S239':
         ocular_projs = [Projs[0]]
         
    if subject == 'S250':
         ocular_projs = [Projs[0]]
    
    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    # data_eeg.plot(events=blinks,show_options=True)
    
    del ocular_projs, blinks, blink_epochs, Projs

#%% Plot data
    reject = dict(eeg=1000e-6)
    epochs = mne.Epochs(data_eeg, data_evnt, [1,2], tmin=-0.3, 
                                     tmax=np.ceil(mseq.size/4096)+0.5,reject=reject, baseline=(-0.1, 0.)) 
    
    epochs_3 = mne.Epochs(data_eeg, data_evnt, [3], tmin=-0.3, 
                                     tmax=np.ceil(mseq.size/4096)+0.5,reject=reject, baseline=(-0.1, 0.)) 
    
    # epochs.average().plot(picks=[31])
    # epochs_3.average().plot(picks=[31])

#%% Extract part of response when stim is on
    ch_picks = np.arange(32)
    remove_chs = []
    
    if subject == 'S207':
        remove_chs = [14]
        
    if subject =='S228':
        remove_chs = [24,27]
        
    ch_picks = np.delete(ch_picks,remove_chs)
    fs = epochs.info['sfreq']
    t = epochs.times
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + mseq.size + int(np.round(0.4*fs))
    
    epdat = epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    epdat_3 = epochs_3.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    t = t[t1:t2]
    t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
    info_obj = epochs.info

#%% Remove epochs with large deflections
    Reject_Thresh=150e-6
    
    if subject =='S207':
        Reject_Thresh = 300e-6
    if subject =='S228':
        Reject_Thresh = 200e-6
    if subject == 'S236':
        Reject_Thresh = 210e-6
    if subject == 'S239':
        Reject_Thresh = 200e-6
    if subject == 'S250':
        Reject_Thresh = 250e-6
    if subject == 'S232':
        Reject_Thresh = 200e-6
    if subject == 'S238':
        Reject_Thresh = 200e-6
    
    Peak2Peak = epdat.max(axis=2) - epdat.min(axis=2)
    Peak2Peak_3 = epdat_3.max(axis=2) -epdat_3.min(axis=2)
    
    mask_trials = np.all(Peak2Peak < Reject_Thresh,axis=0)
    mask_trials_3 = np.all(Peak2Peak_3 < Reject_Thresh,axis=0)
    
    print('rejected ' + str(epdat.shape[1] - sum(mask_trials)) + ' trials due to P2P')
    print('rejected_3v ' + str(epdat_3.shape[1] - sum(mask_trials_3)) + ' trials due to P2P')
    epdat = epdat[:,mask_trials,:]
    epdat_3 = epdat_3[:,mask_trials_3,:]
    print('Total Trials Left: ' + str(epdat.shape[1]))
    print('Total Trials Left_3: ' + str(epdat_3.shape[1]))
    
    Tot_trials = epdat.shape[1] + epdat_3.shape[1]
    
    plt.figure()
    plt.plot(Peak2Peak.T)
    plt.title(subject + ' Tot trials:' + str(Tot_trials))
    
    # plt.figure()
    # plt.plot(Peak2Peak_3.T)

#%% Correlation Analysis

    Ht_12 = mseqXcorr(epdat,mseq[0,:])
    Ht_3 = mseqXcorr(epdat_3,mseq_shift[0,:])
    
    Ht_epochs_12, t_epochs = mseqXcorrEpochs_fft(epdat,mseq[0,:],fs)
    Ht_epochs_3, t_epochs = mseqXcorrEpochs_fft(epdat_3,mseq_shift[0,:],fs)
    
    Ht_epochs = np.concatenate([Ht_epochs_12,Ht_epochs_3],axis=1)
    
    Ht = Ht_12 * epdat.shape[1]/Tot_trials + Ht_3 * epdat_3.shape[1]/Tot_trials



#%% save data
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bit_Active_harder.pickle'),'wb') as file:
        pickle.dump([t, Tot_trials, Ht, info_obj, ch_picks],file)
        
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bit_Active_harder_epochs.pickle'),'wb') as file:
        pickle.dump([Ht_epochs,t_epochs],file)
    
    





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:21:25 2020

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
  
mseq_locs = ['mseqEEG_150_bits7_4096.mat', 'mseqEEG_150_bits8_4096.mat', 
             'mseqEEG_150_bits9_4096.mat', 'mseqEEG_150_bits10_4096.mat']
mseq = []
for m in mseq_locs:
    file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + m
    Mseq_dat = sio.loadmat(file_loc)
    mseq.append( Mseq_dat['mseqEEG_4096'].astype(float) )



#data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S211','S207','S236','S228','S238'] 
num_nfs = 50

for subject in Subjects:
    print('On Subject ...... ' + subject )
    exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    refchans = ['EXG1','EXG2']
    
    if subject == 'S228':
        refchans = ['EXG2']
        
    # exclude = ['EXG1', 'EXG2',
    #            'EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    #refchans = [[None]] #average reference
    
    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=1,h_freq=200)
    
    if subject == 'S211':
        data_eeg.info['bads'].append('A10') #Channel A10 bad in S211
    elif subject == 'S207':
        data_eeg.info['bads'].append('A15') 
        data_eeg.info['bads'].append('A17') 
    elif subject =='S228':
        data_eeg.info['bads'].append('EXG1') 
        data_eeg.info['bads'].append('A30')

   
#%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
    
    ocular_projs = Projs
    if subject ==  'S228':
        ocular_projs = [Projs[0]] #not using Projs[2] for now
    elif subject == 'S211':
        ocular_projs = [Projs[0]] #not using Projs[2] for now
    elif subject == 'S207':
        ocular_projs = [Projs[0]]
    elif subject == 'S236':
        ocular_projs = [Projs[0]] #not using Projs[2] for now
    # elif subject == 'S237':
    #     ocular_projs = [Projs[0]]
    elif subject == 'S238':
        ocular_projs = [Projs[0]] #not using Projs[2] for now
    
    
    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    # data_eeg.plot(events=blinks,show_options=True)
    
    del blinks, blink_epochs, Projs,ocular_projs

#%% Plot data

    labels = ['7bits', '8bits','9bits','10bits']
    reject = dict(eeg=1000e-6)
    epochs = []
    for j in range(len(mseq)):
        epochs.append(mne.Epochs(data_eeg, data_evnt, [j+1], tmin=-0.3, 
                                 tmax=np.ceil(mseq[j].size/4096)+0.4,reject=reject, baseline=(-0.1, 0.)) )
        epochs[j].average().plot(picks=[31],titles = labels[j])
    
    print('On Subject ...... ' + subject )
    
#%% Extract part of response when stim is on
    ch_picks = np.arange(32)
    remove_chs = []
    if subject == 'S211':
        remove_chs = [9] # Ch10 has high P2P
    if subject == 'S207':
        remove_chs = [14,16] #Ch15 has high P2P for many trials
    if subject == 'S228': 
        remove_chs = [29] # High P2P on this forehead channels
    ch_picks = np.delete(ch_picks,remove_chs)
    
    fs = epochs[0].info['sfreq']
    epdat = []
    tdat = []
    for m in range(len(mseq)):
        t = epochs[m].times
        t1 = np.where(t>=0)[0][0]
        t2 = t1 + mseq[m].size + int(np.round(0.4*fs)) #extra 400 ms from end
        epdat.append(epochs[m].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))
        t = t[t1:t2]
        t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
        tdat.append(t)
        
    info_obj = epochs[0].info
    del epochs
#%% Remove epochs with large deflections
    Reject_Thresh=150e-6
    if subject =='S211':  # look at this participant again
        Reject_Thresh = 200e-6
    elif subject == 'S207':
        Reject_Thresh = 250e-6
    elif subject == 'S228':
        Reject_Thresh = 200e-6
    elif subject == 'S236':
        Reject_Thresh = 250e-6
    elif subject == 'S238':
        Reject_Thresh = 250e-6
    #may want to reject a channel in S228
    Tot_trials = np.zeros([len(mseq)])
    for m in range(len(mseq)):
        Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
        mask_trials = np.all(Peak2Peak <Reject_Thresh,axis=0)
        print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
        epdat[m] = epdat[m][:,mask_trials,:]
        print('Total Trials Left: ' + str(epdat[m].shape[1]))
        Tot_trials[m] = epdat[m].shape[1]
        
        plt.figure()
        plt.plot(Peak2Peak.T)
        plt.title(subject + ' ' + str(m))
    
    

#%% Correlation Analysis
    
    Ht = []
    Htnf = []
    Ht_epochs = []
    t_epochs = []
    # do cross corr
    for m in range(len(epdat)): 
        print('On mseq # ' + str(m+1))
        
        Ht_m = mseqXcorr(epdat[m],mseq[m][0,:])
        
        Ht_epochs_m, t_epochs_m = mseqXcorrEpochs_fft(epdat[m],mseq[m][0,:],fs)
        
        Ht.append(Ht_m)
        Ht_epochs.append(Ht_epochs_m)
        t_epochs.append(t_epochs_m)
        
        for nf in range(num_nfs):
            print('On nf: ' + str(nf) )
            resp = epdat[m].copy()
            inv_inds = np.random.permutation(epdat[m].shape[1])[:round(epdat[m].shape[1]/2)]
            resp[:,inv_inds,:] = -resp[:,inv_inds,:]
            Ht_nf = mseqXcorr(resp,mseq[m][0,:])
            Htnf.append(Ht_nf)

    
    #%% Save Data
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4.pickle'),'wb') as file:
        pickle.dump([tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks,],file)
        
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4_epochs.pickle'),'wb') as file:
        pickle.dump([Ht_epochs,t_epochs],file)    
        
    del data_eeg, data_evnt, epdat, tdat, Ht, info_obj,Htnf, Ht_epochs,t_epochs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:52:00 2021

@author: ravinderjit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:51:21 2021

@author: ravinderjit
"""

import numpy as np
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



from sklearn.decomposition import PCA


nchans = 34;
refchans = ['EXG1','EXG2']

mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207','S211','S228','S236','S238','S239','S246','S247','S250']
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    
num_nfs = 1

for subject in Subjects:
    print('On Subject ................... ' + subject)
    
#%% Load data and filter
        
    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=1,h_freq=200)
    
    #%% Remove bad channels
    
    if subject == 'S207':
        data_eeg.info['bads'].append('A15') 
        
    if subject == 'S228':
        data_eeg.info['bads'].append('A24')
    
    
    #%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
    
    if subject == 'S207':
        ocular_projs = [Projs[0]]
    elif subject == 'S228':
        ocular_projs = [Projs[0]]
    elif subject == 'S236':
        ocular_projs = [Projs[0]]
    elif subject == 'S238':
        ocular_projs = [Projs[0]] #Projs[1]
    elif subject == 'S239':
        ocular_projs = [Projs[0]] # Projs[2
    elif subject == 'S246':
        ocular_projs = [Projs[0]] #Projs[1]
    elif subject == 'S247':
        ocular_projs = [Projs[0]] #Projs[2]
    elif subject == 'S250':
        ocular_projs = [Projs[0]]  #Projs[1]
    elif subject == 'S211':
        ocular_projs = [Projs[0]] #Projs[2]
    
    
    data_eeg.add_proj(ocular_projs)
    data_eeg.plot_projs_topomap()
    # data_eeg.plot(events=blinks,show_options=True)
    
    del blinks, blink_epochs, Projs,ocular_projs
    
    #%% Plot data
    fs = data_eeg.info['sfreq']
    reject = dict(eeg=1000e-6)
    epochs = mne.Epochs(data_eeg,data_evnt,1,tmin=-0.3,tmax=np.ceil(mseq.size/fs)+0.4, reject = reject, baseline=(-0.1,0.))     
    epochs.average().plot(picks=[31],titles='10 bit AMmseq')
    
    #%% Extract part of response when stim is on
    ch_picks = np.arange(32)
    remove_chs = []
    
    if subject == 'S207':
        remove_chs = [14]
    elif subject == 'S228':
        remove_chs = [23]
    
    ch_picks = np.delete(ch_picks,remove_chs)
    
    t = epochs.times
    t1 = np.where(t>=0)[0][0]
    t2 = t1 + mseq.size + int(np.round(0.4*fs))
    epdat = epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
    t = t[t1:t2]
    t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
    
    info_obj = epochs[0].info
    #del epochs
    #%% Remove epochs with large deflections
    Reject_Thresh = 150e-6
    if subject == 'S207':
        Reject_Thresh = 300e-6
    elif subject == 'S228':
        Reject_Thresh = 300e-6
    elif subject == 'S236':
        Reject_Thresh = 300e-6
    elif subject == 'S239':
        Reject_Thresh = 300e-6
    elif subject == 'S246':
        Reject_Thresh = 250e-6
    elif subject == 'S247':
        Reject_Thresh = 250e-6
    elif subject == 'S250':
        Reject_Thresh = 200e-6
    elif subject == 'S246':
        Reject_Thresh = 250e-6
    elif subject == 'S238':
        Reject_Thresh = 200e-6
    
    Peak2Peak = epdat.max(axis=2) - epdat.min(axis=2)
    mask_trials = np.all(Peak2Peak < Reject_Thresh,axis=0)
    print('rejected ' + str(epdat.shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat = epdat[:,mask_trials,:]
    print('Total Trials Left: ' + str(epdat.shape[1]))
    Tot_trials = epdat.shape[1]
    
    ch_maxAmp = Peak2Peak.mean(axis=1).argmax()
    plt.figure()
    plt.plot(Peak2Peak.T)
    plt.title(subject + ' Tot_trials: ' + str(Tot_trials))
    
    # plt.figure()
    # plt.plot(Peak2Peak[:,mask_trials].T)
    
    # plt.figure()
    # plt.plot(Peak2Peak[np.delete(ch_picks,ch_maxAmp),:].T)
    
    #%% Correlation Analysis
    
    Htnf = []
    
    Ht = mseqXcorr(epdat,mseq[0,:])
    Ht_epochs, t_epochs = mseqXcorrEpochs_fft(epdat,mseq[0,:],fs)
    
    for nf in range(num_nfs):
        print('On nf:' + str(nf))
        resp = epdat.copy()
        inv_inds = np.random.permutation(epdat.shape[1])[:round(epdat.shape[1]/2)]
        resp[:,inv_inds,:] = -resp[:,inv_inds,:]
        Ht_nf = mseqXcorr(resp,mseq[0,:])
        Htnf.append(Ht_nf)
  
    
    #%% Save Data
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bits_Active.pickle'),'wb') as file:
        pickle.dump([t, Tot_trials, Ht, Htnf, info_obj, ch_picks],file)
        
    with open(os.path.join(pickle_loc,subject+'_AMmseq10bits_Active_epochs.pickle'),'wb') as file:
        pickle.dump([Ht_epochs, t_epochs],file)
    del data_eeg, data_evnt, epdat, t, Ht, info_obj, Htnf
    
    
    
    
     
    
        
        
        
        
        
        
        
        
        
        

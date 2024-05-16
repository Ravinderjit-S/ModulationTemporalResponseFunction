#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:15:54 2021

@author: ravinderjit
Investigate repeatability of "ACR"
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import scipy.io as sio
from scipy.signal import find_peaks

import sys
sys.path.append(os.path.abspath('../ACRanalysis/'))
from ACR_helperFuncs import ACR_sourceHf
from ACR_helperFuncs import Template_tcuts
from ACR_helperFuncs import PCA_tcuts
from ACR_helperFuncs import PCA_tcuts_topomap


#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)


#%% Load Template
template_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/Pickles/PCA_passive_template.pickle'

with open(template_loc,'rb') as file:
    [pca_coeffs_cuts,pca_expVar_cuts,t_cuts] = pickle.load(file)
    
bstemTemplate  = pca_coeffs_cuts[0][0,:]
cortexTemplate = pca_coeffs_cuts[2][0,:]

# vmin = bstemTemplate.mean() - 2 * bstemTemplate.std()
# vmax = bstemTemplate.mean() + 2 * bstemTemplate.std()
# plt.figure()
# mne.viz.plot_topomap(cortexTemplate, mne.pick_info(info_obj,np.arange(32)),vmin=vmin,vmax=vmax)
        



#%% Load data collected first

data_loc_old = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc_old = data_loc_old + 'Pickles/'
Subjects = ['S207','S228','S236','S238'] 

A_Tot_trials_old = []
A_Ht_old = []
A_Htnf_old = []
A_info_obj_old = []
A_ch_picks_old = []

A_Ht_old_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4.pickle'),'rb') as file:
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials[3])
    A_Ht_old.append(Ht[3])
    A_Htnf_old.append(Htnf[3])
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        A_Ht_old_epochs.append(Ht_epochs[3])
    
t = tdat[3]


#%% Add first run with 10bit stim
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

Subjects.append('S250')
subject = 'S250'
with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
    [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials)
    A_Ht_old.append(Ht)
    A_Htnf_old.append(Htnf)
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
    [Ht_epochs, t_epochs] = pickle.load(file)
    A_Ht_old_epochs.append(Ht_epochs)


print('Done loading 1st visit ...')
#%% Load Second Run of data collection
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

A_Ht_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs.append(Ht_epochs)
    

print('Done loading 2nd visit ...')



#%% Example CZ
sub = 0
ch_cz = np.where(A_ch_picks[sub]==31)[0][0]
cz = A_Ht[sub][ch_cz,:]

plt.figure()
plt.plot(t*1000,cz, color='k',linewidth = 2)
plt.xlim([-100,500])
#plt.xticks([7.3,29, 47, 94, 201, 500, 1000],labels=['7.3','29','47','94','201','500','1000'])
plt.xlabel('Time (msec)', fontsize=12)
plt.ylabel('Amplitude',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('mod-TRF Ch. Cz',fontsize=14)
#plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#plt.xscale('log')

#%% Plot Ch. Cz & FP1
    
for sub in range(len(Subjects)):
    plt.figure()
    ch_old_cz = np.where(A_ch_picks_old[sub]==31)[0][0]
    ch_cz = np.where(A_ch_picks[sub]==31)[0][0]
    
    ch_old_fp = np.where(A_ch_picks_old[sub]==0)[0][0]
    ch_fp = np.where(A_ch_picks[sub]==0)[0][0]
    
    cz_1 = A_Ht_old[sub][ch_old_cz,:]
    cz_2 = A_Ht[sub][ch_cz,:]
    
    [troughs_cz1,properties] = find_peaks(-cz_1)
    [peaks_cz1, properties] = find_peaks(cz_1)
    
    fp_1 = -A_Ht_old[sub][ch_old_fp,:]
    fp_2 = -A_Ht[sub][ch_fp,:]
    
    [troughs_fp1, properties] = find_peaks(-fp_1)
    [peaks_fp1, properties] = find_peaks(fp_1)
    
    plt.plot(t,cz_1, label='Visit 1', color='tab:blue')
    plt.plot(t,cz_2, label='Visit 2', color='tab:orange')
    # plt.plot(t,fp_1, label='Visit 1', color='tab:blue',linestyle='dashed')
    # plt.plot(t,fp_2, label='Visit 2', color='tab:orange',linestyle='dashed')
    
    # plt.scatter(t[troughs_cz1],cz_1[troughs_cz1],color='blue')
    # plt.scatter(t[peaks_cz1],cz_1[peaks_cz1],color='blue')
    
    # plt.scatter(t[troughs_fp1],fp_1[troughs_fp1],color='blue')
    # plt.scatter(t[peaks_fp1],fp_1[peaks_fp1],color='blue')
    
    
    plt.title(Subjects[sub])
    plt.legend()
    plt.xlim([0,0.5])
    
    
#%% Plot 32 channel repeat

sbp = [4,4]
sbp2 = [4,4]

for sub in range(len(Subjects)):
    ch_picks_old_s = A_ch_picks_old[sub] 
    ch_picks_s = A_ch_picks[sub]
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks_old_s):
                ch_ind = np.where(cur_ch == ch_picks_old_s)[0][0]
                axs[p1,p2].plot(t,A_Ht_old[sub][ch_ind,:])
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch == ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
            
            axs[p1,p2].set_xlim([-0.1,0.5])
            axs[p1,p2].set_title('A' + str(ch_picks[ch_ind]+1))
            
    fig.suptitle(Subjects[sub])
            

for sub in range(len(Subjects)):
    ch_picks_old_s = A_ch_picks_old[sub] 
    ch_picks_s = A_ch_picks[sub]
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_old_s):
                ch_ind = np.where(cur_ch == ch_picks_old_s)[0][0]
                axs[p1,p2].plot(t,A_Ht_old[sub][ch_ind,:])
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch == ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
            
            axs[p1,p2].set_xlim([-0.1,0.5])
            axs[p1,p2].set_title('A' + str(ch_picks[ch_ind]+1))
            
    fig.suptitle(Subjects[sub])


#%% Plot template responses
    
for sub in range(len(Subjects)):
    bs_resp_old = np.matmul(A_Ht_old[sub].T,bstemTemplate[A_ch_picks_old[sub]])
    bs_resp_2 = np.matmul(A_Ht[sub].T,bstemTemplate[A_ch_picks[sub]])
    
    ctx_resp_old = np.matmul(A_Ht_old[sub].T,cortexTemplate[A_ch_picks_old[sub]])
    ctx_resp_2 = np.matmul(A_Ht[sub].T,cortexTemplate[A_ch_picks[sub]])

    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,bs_resp_old,label='Visit 1')
    ax[0].plot(t,bs_resp_2, label = 'Visit 2')
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_title(Subjects[sub] + ' Brainstem')
    ax[0].set_xlim([0,0.5])
    ax[0].legend()
    
    ax[1].plot(t,ctx_resp_old, label= 'Visit 1')
    ax[1].plot(t,ctx_resp_2, label='Visit 2')
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_title('Cortex')
    ax[1].set_xlim([0,0.5])


#%% Manual Peak Cutoff Reading CZ and FP
    
cuts_tms = []
#S207
cuts_tms.append([.0185, .036, .067, .136 ,.266])

#S228
cuts_tms.append([.016, .043, .069, .125, .238])

#S236
cuts_tms.append([.014, .031, .066, .124, .249])

#238
cuts_tms.append([.0155, .045, .062, .124, .287])

#S250
cuts_tms.append([.016, .049, .123, .220, .334])

#%% Compute Response from tcuts templates
colors = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
bs_sub_expVar = []
ctx_sub_expVar = []

bs_sub_expVar_2 = []
ctx_sub_expVar_2 = []

for sub in range(len(Subjects)):
    t_tc, bs_tc_sp, bs_tc_expVar = Template_tcuts(A_Ht_old[sub], t, cuts_tms[sub], A_ch_picks_old[sub], bstemTemplate)
    t_tc, ctx_tc_sp, ctx_tc_expVar = Template_tcuts(A_Ht_old[sub], t, cuts_tms[sub], A_ch_picks_old[sub], cortexTemplate)
    
    t_tc, bs_tc_sp_2, bs_tc_expVar_2 = Template_tcuts(A_Ht[sub], t, cuts_tms[sub], A_ch_picks[sub], bstemTemplate)
    t_tc, ctx_tc_sp_2, ctx_tc_expVar_2 = Template_tcuts(A_Ht[sub], t, cuts_tms[sub], A_ch_picks[sub], cortexTemplate)
    
    bs_sub_expVar.append(bs_tc_expVar)
    ctx_sub_expVar.append(ctx_tc_expVar)
    
    bs_sub_expVar_2.append(bs_tc_expVar_2)
    ctx_sub_expVar_2.append(ctx_tc_expVar_2)
    
    fig,ax = plt.subplots(2,1)
    for tc in range(len(t_tc)):
        ax[0].plot(t_tc[tc],bs_tc_sp[tc],label='Visit 1',color=colors[tc])
        ax[0].plot(t_tc[tc],bs_tc_sp_2[tc],label='Visit 2',linestyle='dashed',color=colors[tc])
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_title(Subjects[sub] + ' Brainstem')
        ax[0].set_xlim([0,0.5])
        #ax[0].legend()
        
        ax[1].plot(t_tc[tc],ctx_tc_sp[tc], label= 'Visit 1',color=colors[tc])
        ax[1].plot(t_tc[tc],ctx_tc_sp_2[tc], label= 'Visit 2',linestyle='dashed',color=colors[tc])
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_title('Cortex')
        ax[1].set_xlim([0,0.5])
    

#%% Compute response from individual templates

for sub in range(len(Subjects)):
    pca_sp_cuts_, pca_expVar_cuts_, pca_coeff_cuts_, t_cuts_ = PCA_tcuts(
        A_Ht_old[sub], t, cuts_tms[sub], A_ch_picks_old[sub], np.arange(A_ch_picks_old[sub].size))
    
    PCA_tcuts_topomap(pca_coeff_cuts_, cuts_tms[sub] , pca_expVar_cuts_, A_info_obj_old[sub], A_ch_picks_old[sub], Subjects[sub] + ' Visit 1')
    
    pca_sp_cuts_, pca_expVar_cuts_, pca_coeff_cuts_, t_cuts_ = PCA_tcuts(
    A_Ht[sub], t, cuts_tms[sub], A_ch_picks[sub], np.arange(A_ch_picks[sub].size))

        
    PCA_tcuts_topomap(pca_coeff_cuts_, cuts_tms[sub] , pca_expVar_cuts_, A_info_obj[sub], A_ch_picks[sub], Subjects[sub] + ' Visit 2')
    
#... gonna go with average templates for now
    
#%% Plot Cz and Fp1 using tcuts
    
axs = []
for sub in range(len(Subjects)):
    #plt.figure()
    ch_old_cz = np.where(A_ch_picks_old[sub]==31)[0][0]
    ch_cz = np.where(A_ch_picks[sub]==31)[0][0]
    
    ch_old_fp = np.where(A_ch_picks_old[sub]==0)[0][0]
    ch_fp = np.where(A_ch_picks[sub]==0)[0][0]
    
    cz_1 = A_Ht_old[sub][ch_old_cz,:]
    cz_2 = A_Ht[sub][ch_cz,:]
    
    
    fp_1 = -A_Ht_old[sub][ch_old_fp,:]
    fp_2 = -A_Ht[sub][ch_fp,:]
    
    t_cuts = cuts_tms[sub]
    
    
    for t_c in range(len(t_tc)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        # plt.plot(t[t_1:t_2],cz_1[t_1:t_2] - cz_1[t_1:t_2].mean(), label='Visit 1', color=colors[t_c])
        # plt.plot(t[t_1:t_2],cz_2[t_1:t_2] - cz_2[t_1:t_2].mean() , label='Visit 2', color=colors[t_c],linestyle='dashed')
        
        ax = plt.subplot(2,3,sub+1)
        ax.plot(t[t_1:t_2],cz_1[t_1:t_2] - cz_1[t_1:t_2].mean(), label='Visit 1', color=colors[t_c])
        #ax.plot(t[t_1:t_2],cz_2[t_1:t_2] - cz_2[t_1:t_2].mean() , label='Visit 2', color=colors[t_c],linestyle='dashed')
        
        # plt.plot(t_tc[tc],fp_1, label='Visit 1', color='tab:blue',linestyle='dashed')
        # plt.plot(t_tc[tc],fp_2, label='Visit 2', color='tab:orange',linestyle='dashed')
    
    
    
    # plt.title(Subjects[sub] + ' Ch. Cz',fontsize=12)
    # #plt.legend()
    # plt.xlim([0,0.4])
    # plt.xticks([0,0.1,0.2,0.3,0.4],fontsize=10)
    # plt.yticks([-.0015,.0015],fontsize=10)
    # plt.xlabel('Time (sec)',fontsize=12)
    # plt.ylabel('Amplitdue',fontsize=12)
    # plt.axes().ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    
    ax.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    ax.set_xticks([0,0.2,0.4])
    ax.set_yticks([-.002,0,.002])
    ax.set_xlim([0,0.4])
    ax.set_title(Subjects[sub],fontweight='bold')
    
    axs.append(ax)
    
axs[3].set_xlabel('Time (s)',fontweight='bold')
axs[3].set_ylabel('Amplitdue',fontweight='bold')    

#%% Look at epochs: 1st visit and 2nd Visit

fig = plt.figure()
ax = [None] *5
ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

for sub in np.arange(len(Subjects)):
    #Plot Cz
    
    if sub > 0:
        ax[sub].axes.yaxis.set_visible(False)

        
    Ht_mean_old = A_Ht_old_epochs[sub].mean(axis=1) 
    Ht_mean = A_Ht_epochs[sub].mean(axis=1)
    
    Ht_sem_old = A_Ht_old_epochs[sub].std(axis=1) / np.sqrt(A_Ht_old_epochs[sub].shape[1])
    Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1])
    
    ax[sub].plot(t_epochs, Ht_mean[-1,:],color='grey', label='2nd Visit')
    ax[sub].fill_between(t_epochs,Ht_mean[-1,:] -Ht_sem[-1,:],Ht_mean[-1,:] + Ht_sem[-1,:], color='grey',alpha=0.5)
    
    ax[sub].plot(t_epochs, Ht_mean_old[-1,:],color='k',label='1st Visit')
    ax[sub].fill_between(t_epochs,Ht_mean_old[-1,:] -Ht_sem_old[-1,:],Ht_mean_old[-1,:] + Ht_sem_old[-1,:], color='k',alpha=0.5)
    
    t_cuts = cuts_tms[sub]
    
    # for t_c in range(len(t_cuts)):
    #     if t_c ==0:
    #         t_1 = np.where(t_epochs>=0)[0][0]
    #     else:
    #         t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
    #     t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
    #     ax[sub].plot(t_epochs[t_1:t_2], Ht_mean_old[-1,t_1:t_2],color=colors[t_c],label='Source ' + str(t_c))
    #     ax[sub].fill_between(t_epochs[t_1:t_2],Ht_mean_old[-1,t_1:t_2] -Ht_sem_old[-1,t_1:t_2],Ht_mean_old[-1,t_1:t_2] + Ht_sem_old[-1,t_1:t_2], color=colors[t_c],alpha=0.5)
        

    ax[sub].set_xlim([-0.1,0.5])
    ax[sub].set_title('Subject ' + str(sub+1),fontweight='bold')
    ax[sub].set_xticks([0,0.1,0.2,0.3,0.4])


ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Amplitude')
ax[0].set_yticks([-.002,0,.002,.004])
ax[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[0].legend(fontsize=9)
fig.suptitle('Ch. Cz',fontweight='bold')

#%% Look at epochs with t cuts and in freq domain

fs = 4096
fig = plt.figure()
ax = [None] *5
ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

fig_f = plt.figure()
axf = [None] *5
axf[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
axf[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
axf[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
axf[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
axf[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

for sub in np.arange(len(Subjects)):
    #Plot Cz
    
    if sub > 0:
        ax[sub].axes.yaxis.set_visible(False)
        axf[sub].axes.yaxis.set_visible(False)
        
    Ht_mean = A_Ht_epochs[sub].mean(axis=1)
    Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1])
    

    t_cuts = cuts_tms[sub]
    for t_c in range(len(t_tc)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        t_ep = t_epochs[t_1:t_2]
        Ht_mean_tc = Ht_mean[-1,t_1:t_2] #- Ht_mean[-1,t_1:t_2].mean()
        
        Ht_freq = np.abs(np.fft.fft(A_Ht_epochs[sub][-1,:,t_1:t_2] - A_Ht_epochs[sub][-1,:,t_1:t_2].mean(axis=-1)[:,np.newaxis]))
        Ht_freq_mean = Ht_freq.mean(axis=0)
        Ht_freq_sem = Ht_freq.std(axis=0) / np.sqrt(Ht_freq.shape[0])     
                   
        f = np.fft.fftfreq(Ht_freq_mean.size,d=1/fs)

        ax[sub].plot(t_ep, Ht_mean_tc, color=colors[t_c])
        ax[sub].fill_between(t_ep,Ht_mean_tc -Ht_sem[-1,t_1:t_2],Ht_mean_tc + Ht_sem[-1,t_1:t_2], color=colors[t_c],alpha=0.5)
        ax[sub].set_title('Subject ' + str(sub+1))
        ax[sub].set_xticks([0,0.1,0.2,0.3,0.4])
        
        axf[sub].plot(f,Ht_freq_mean,color=colors[t_c])
        axf[sub].fill_between(f,Ht_freq_mean-Ht_freq_sem,Ht_freq_mean+Ht_freq_sem,color=colors[t_c],alpha=0.5)
        axf[sub].set_title('Subject ' + str(sub+1))
        axf[sub].set_xlim([0,75])
        axf[sub].set_xticks([10,25,50])
        


ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Amplitude')
ax[0].set_yticks([-.002,0,.002,.004])
ax[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#ax[0].legend(fontsize=9)
#fig.suptitle('Ch. Cz',fontweight='bold')

axf[0].set_xlabel('Frequency (Hz)')
axf[0].set_ylabel('Magnitude')
axf[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#%% 
colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown']





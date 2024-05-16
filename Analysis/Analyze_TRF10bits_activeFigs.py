#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:12:16 2022

@author: ravinderjit

make figues to show what happens to mod-TRF under attention

"""


import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio


#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
        
#%% Subjects

Subjects = ['S207', 'S211', 'S228','S236','S238','S239','S246','S247','S250']
Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now


dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

dataCount_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickleCount_loc = dataCount_loc + 'Pickles/'

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')


#%% Load Passive Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    print('Loading ' + Subjects[sub])
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs_pass.append(Ht_epochs)

print('Done loading passive ...')

#%% Load Counting Data

A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

A_Ht_epochs_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    print ('loading ' + subject)
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_count.append(Ht_epochs)
    
print('Done Loading Counting data')

#%% Load Shift Detect Data

Subjects_sd = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250'] 

A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

A_Ht_epochs_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    print('loading ' + subject)
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)
    
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_sd.append(Ht_epochs)
    
print('Done loading shift detect data')

#%% Plot Ch. Cz
    
t_epochs *= 1e3 #change to ms
fig,ax = plt.subplots(nrows=2,ncols=2,sharex=True)
fig.set_size_inches(12,8)
ax = np.reshape(ax,4)

t_0 = np.where(t_epochs>=0)[0][0]

for sub in range(len(Subjects[:4])):
    subject = Subjects[sub]
    
    ch = 31
    
    #if sub !=2:
        #ax[sub].axes.yaxis.set_visible(False)
        
    # if sub < len(Subjects)/2:
    #     ax[sub].axes.xaxis.set_visible(False)
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_epochs_pass[sub][ch_pass_ind,:,:] *1e6
    cz_count = A_Ht_epochs_count[sub][ch_count_ind,:,:] * 1e6
    
    cz_pass_sem = cz_pass.std(axis=0) / np.sqrt(cz_pass.shape[0])
    cz_count_sem = cz_count.std(axis=0) / np.sqrt(cz_count.shape[0])
    
    cz_pass = cz_pass.mean(axis=0)
    cz_count = cz_count.mean(axis=0)
    
    cz_pass = cz_pass - cz_pass[t_0] #make time 0 have value of 0
    cz_count = cz_count - cz_count[t_0] #make time 0 have value of 0
    
    ax[sub].plot(t_epochs,cz_pass, label='Passive', color='k',linewidth=2)
    ax[sub].fill_between(t_epochs,cz_pass-cz_pass_sem, cz_pass+cz_pass_sem,color='k',alpha=0.5)
    
    ax[sub].plot(t_epochs,cz_count, label='Count (easy)', color='tab:blue')
    ax[sub].fill_between(t_epochs,cz_count-cz_count_sem, cz_count+cz_count_sem,color='tab:blue',alpha=0.5)
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_epochs_sd[sub_sd][ch_sd_ind,:,:] * 1e6
        
        cz_sd_sem = cz_sd.std(axis=0) / np.sqrt(cz_sd.shape[0])
        cz_sd = cz_sd.mean(axis=0)
        
        cz_sd = cz_sd - cz_sd[t_0]
        
        ax[sub].plot(t_epochs,cz_sd, label='Shift Detect (hard)', color='tab:orange')
        ax[sub].fill_between(t_epochs,cz_sd - cz_sd_sem, cz_sd + cz_sd_sem, color='tab:orange',alpha=0.5)
    
    #ax[sub].set_title('S' + str(sub+1))
    ax[sub].set_title('P' + str(sub+1),fontweight='bold',fontsize=14)
    ax[sub].set_xlim([-0.010*1e3,0.3*1e3])
    #ax[sub].set_xlim([-.005,0.050])
    #ax[sub].set_xticks([0,0.050,0.1])
    #ax[sub].set_xticks([0,0.2,0.4])
    
ax[2].legend(fontsize=12)
#ax[2].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[2].set_xlabel('Time (msec)',fontsize=14)
ax[2].set_ylabel('\u03BCV',fontsize=14)
ax[0].set_yticks([-15,0,15,30])
ax[1].set_yticks([-20,-10,0,10])
ax[2].set_yticks([-30,0,30,60])
ax[3].set_yticks([-40,-20,0,20])
for a_ in ax:
    a_.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(fig_path,'ModTRF_active.svg'),format='svg')

t_epochs /= 1e3

#%% Average Attention Effects

# Average Passive Data

t_neg = np.where(t>=-0.007)[0][0] # baseline from the last 7 ms
t_0 = np.where(t>=0)[0][0]

#32 ch average
perCh_pass = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh_pass[ch,0] += np.sum(A_ch_picks_pass[s]==ch)

Avg_Ht_pass = np.zeros([32,t.size])
Ht_pass_cz = np.zeros([len(Subjects),t.size])

for s in range(len(Subjects)):
    Ht_s = A_Ht_pass[s] - A_Ht_pass[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_pass[A_ch_picks_pass[s],:] += Ht_s
    Ht_pass_cz[s,:] = Ht_s[-1,:]

Avg_Ht_pass = Avg_Ht_pass / perCh_pass

# Average Counting Data
#32 ch average
perCh_count = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh_count[ch,0] += np.sum(A_ch_picks_count[s]==ch)

Avg_Ht_count = np.zeros([32,t.size])
Ht_count_cz = np.zeros([len(Subjects),t.size])

for s in range(len(Subjects)):
    Ht_s = A_Ht_count[s] - A_Ht_count[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_count[A_ch_picks_count[s],:] += Ht_s
    Ht_count_cz[s,:] = Ht_s[-1,:]

Avg_Ht_count = Avg_Ht_count / perCh_count

# Average Shift Detect Data 
#32 ch average
perCh_sd = np.zeros([32,1])
for s in range(len(Subjects_sd)):
    for ch in range(32):
        perCh_sd[ch,0] += np.sum(A_ch_picks_sd[s]==ch)

Avg_Ht_sd = np.zeros([32,t.size])
Ht_sd_cz = np.zeros([len(Subjects_sd),t.size])

for s in range(len(Subjects_sd)):
    Ht_s = A_Ht_sd[s] - A_Ht_sd[s][:,t_neg:t_0].mean(axis=1)[:,np.newaxis] #make time zero = 0
    Avg_Ht_sd[A_ch_picks_sd[s],:] += Ht_s
    Ht_sd_cz[s,:] = Ht_s[-1,:]

Avg_Ht_sd = Avg_Ht_sd / perCh_sd

plt.figure()
plt.plot(t, Avg_Ht_pass[31,:])
plt.plot(t, Avg_Ht_count[31,:])
plt.plot(t, Avg_Ht_sd[31,:])
plt.xlim([-0.05,0.4])

pass_mean = Ht_pass_cz.mean(axis=0) * 1e6
count_mean = Ht_count_cz.mean(axis=0)* 1e6
sd_mean = Ht_sd_cz.mean(axis=0)* 1e6

pass_sem = Ht_pass_cz.std(axis=0) / np.sqrt(Ht_pass_cz.shape[0])* 1e6
count_sem = Ht_count_cz.std(axis=0) / np.sqrt(Ht_count_cz.shape[0])* 1e6
sd_sem = Ht_sd_cz.std(axis=0) / np.sqrt(Ht_sd_cz.shape[0])* 1e6

fig = plt.figure()
t*=1e3
fig.set_size_inches(10,5)
plt.plot(t,pass_mean,color='k',linewidth=2, label='Passive')
plt.fill_between(t,pass_mean - pass_sem, pass_mean+pass_sem,color='k',alpha=0.5)

plt.plot(t,count_mean, color='tab:blue',linewidth=2, label='Easy (Counting)')
plt.fill_between(t,count_mean - count_sem, count_mean+pass_sem,color='tab:blue',alpha=0.5)

plt.plot(t, sd_mean, color='tab:orange',linewidth=2, label='Hard (Shift Detect)')
plt.fill_between(t,sd_mean - sd_sem, sd_mean+ sd_sem,color='tab:orange',alpha=0.5)
t/=1e3
plt.xlim(-50,300)
#plt.xticks([0,50,100,200,300])
plt.yticks([-15,0,15,30])
plt.ylabel('\u03BCV',fontsize=14)
plt.xlabel('Time (msec)',fontsize=14)
plt.tick_params(labelsize=14)
plt.legend(fontsize=12)

plt.savefig(os.path.join(fig_path,'ModTRF_active_avg.svg'),format='svg')

#Same Figure but zoom in on first 50 ms

fig = plt.figure()
t*=1e3
fig.set_size_inches(12,6)
plt.plot(t,pass_mean,color='k',linewidth=2, label='Passive')
plt.fill_between(t,pass_mean - pass_sem, pass_mean+pass_sem,color='k',alpha=0.5)

plt.plot(t,count_mean, color='tab:blue',linewidth=2, label='Easy (Counting)')
plt.fill_between(t,count_mean - count_sem, count_mean+pass_sem,color='tab:blue',alpha=0.5)

plt.plot(t, sd_mean, color='tab:orange',linewidth=2, label='Hard (Shift Detect)')
plt.fill_between(t,sd_mean - sd_sem, sd_mean+ sd_sem,color='tab:orange',alpha=0.5)
t/=1e3
plt.xlim(-10,50)
#plt.xticks([0,50,100,200,300])
plt.yticks([-15,0,15,30])
plt.ylabel('\u03BCV',fontsize=18)
plt.xlabel('Time (msec)',fontsize=18)
plt.tick_params(labelsize=18)
#plt.legend()

plt.savefig(os.path.join(fig_path,'ModTRF_active_avg_1st50.svg'),format='svg')



#%% PCA on Average Attention Effects to get rough sources


t_cuts_pass = [.016, .033, .066, .123, .268 ]
t_cuts_count = [.015, .037, .068, .120, .260  ]
t_cuts_sd = [.013, .036, .066, .123, .260 ]


pca_sp_pass = []
pca_expVar_pass = []
pca_coeff_pass = []
t_cutT_pass = []

pca_sp_count = []
pca_expVar_count = []
pca_coeff_count = []
t_cutT_count = []

pca_sp_sd = []
pca_expVar_sd = []
pca_coeff_sd = []
t_cutT_sd = []

pca = PCA(n_components=1)

for t_c in range(len(t_cuts_pass)):
    
    # Get times for t_cut
    if t_c ==0:
        t_1_p = np.where(t>=0)[0][0]
        t_1_c = t_1_p
        t_1_s = t_1_p
    else:
        t_1_p = np.where(t>=t_cuts_pass[t_c-1])[0][0]
        t_1_c = np.where(t>=t_cuts_count[t_c-1])[0][0]
        t_1_s = np.where(t>=t_cuts_sd[t_c-1])[0][0]
         
    t_2_p = np.where(t>=t_cuts_pass[t_c])[0][0]
    t_2_c = np.where(t>=t_cuts_count[t_c])[0][0]
    t_2_s = np.where(t>=t_cuts_sd[t_c])[0][0]
    
    #PCA
    pca_sp_pass_tc = pca.fit_transform(Avg_Ht_pass[:,t_1_p:t_2_p].T)
    pca_expVar_pass_tc = pca.explained_variance_ratio_
    pca_coeff_pass_tc = pca.components_
    
    pca_sp_count_tc = pca.fit_transform(Avg_Ht_count[:,t_1_p:t_2_p].T)
    pca_expVar_count_tc = pca.explained_variance_ratio_
    pca_coeff_count_tc = pca.components_
    
    pca_sp_sd_tc = pca.fit_transform(Avg_Ht_sd[:,t_1_p:t_2_p].T)
    pca_expVar_sd_tc = pca.explained_variance_ratio_
    pca_coeff_sd_tc = pca.components_
    
    
    # Get polarity with positive at Cz
    if pca_coeff_pass_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_pass_tc = -pca_coeff_pass_tc
       pca_sp_pass_tc = -pca_sp_pass_tc
       
    if pca_coeff_count_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_count_tc = -pca_coeff_count_tc
       pca_sp_count_tc = -pca_sp_count_tc
    
    if pca_coeff_sd_tc[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff_sd_tc = -pca_coeff_sd_tc
       pca_sp_sd_tc = -pca_sp_sd_tc
       
    #Store in lists
    pca_sp_pass.append(pca_sp_pass_tc)
    pca_expVar_pass.append(pca_expVar_pass_tc)
    pca_coeff_pass.append(pca_coeff_pass_tc)
    t_cutT_pass.append(t[t_1_p:t_2_p])
    
    pca_sp_count.append(pca_sp_count_tc)
    pca_expVar_count.append(pca_expVar_count_tc)
    pca_coeff_count.append(pca_coeff_count_tc)
    t_cutT_count.append(t[t_1_p:t_2_p])
    
    pca_sp_sd.append(pca_sp_sd_tc)
    pca_expVar_sd.append(pca_expVar_sd_tc)
    pca_coeff_sd.append(pca_coeff_sd_tc)
    t_cutT_sd.append(t[t_1_p:t_2_p])



plt.figure()
for t_c in range(len(t_cuts_pass)):
    plt.plot(t_cutT_pass[t_c],pca_sp_pass[t_c][:,0])
plt.plot(t,Avg_Ht_pass[31,:] -Avg_Ht_pass[31,:].mean(),color='k')
plt.xlim([0,0.5])
    

fig = plt.figure()
fig.set_size_inches(10,5)
labels = ['comp1']
vmin = pca_coeff_pass[-1].mean() - 2 * pca_coeff_pass[-1].std()
vmax = pca_coeff_pass[-1].mean() + 2 * pca_coeff_pass[-1].std()
for t_c in range(len(t_cuts_pass)):
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1)
    plt.title( str(np.round(pca_expVar_pass[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_pass[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1 + len(t_cuts_pass))
    plt.title(str(np.round(pca_expVar_count[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_count[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
    ax = plt.subplot(3,len(t_cuts_pass),t_c+1 + 2* len(t_cuts_pass))
    plt.title( str(np.round(pca_expVar_sd[t_c][0]*100)) + '%',horizontalalignment='right')
    mne.viz.plot_topomap(pca_coeff_sd[t_c][0,:], mne.pick_info(info_obj,A_ch_picks_pass[1]),vlim=(vmin,vmax),axes=ax)
    
plt.savefig(os.path.join(fig_path,'topomaps.svg'),format='svg')







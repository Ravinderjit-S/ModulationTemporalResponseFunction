#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:44:40 2021

@author: ravinderjit
"""


import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#%% Subjects

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')

Subjects = ['S207', 'S211', 'S228','S236','S238','S239','S246','S247','S250']

dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

#%% Load Data

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

A_Ht_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    print('Loading ' + subject)
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs.append(Ht_epochs)

print('Done loading passive ...')


#%% Average

Cz_sub = []

for sub in range(len(Subjects)):
    Cz_sub.append(A_Ht[sub][-1,:])

Cz_sub = np.array(Cz_sub) * 1e6 #untis of microvolts
Cz_mean = Cz_sub.mean(axis=0)
Cz_sem = Cz_sub.std(axis=0) / np.sqrt(Cz_sub.shape[0])

plt.figure()
plt.plot(t,Cz_mean,color='k',linewidth=2)
plt.fill_between(t,Cz_mean-Cz_sem,Cz_mean+Cz_sem,color='k',alpha=0.5)
plt.xlim([-0.05,0.4])
plt.xlabel('Time (sec)')
plt.ylabel('\u03BCV')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
plt.ticklabel_format(axis='y') 
plt.title('Average mod-TRF across 9 Particpants')


# average with t_cuts
t_cuts = [.016, .033, .066, .123, .268 ]
colors=  ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown']

fig =plt.figure()
fig.set_size_inches(8,4.5)

for tc in range(len(t_cuts)):
    if tc ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[tc-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[tc])[0][0]
    plt.plot(t[t_1:t_2],Cz_mean[t_1:t_2],color=colors[tc],linewidth=2)
    plt.fill_between(t[t_1:t_2],Cz_mean[t_1:t_2]-Cz_sem[t_1:t_2],Cz_mean[t_1:t_2]+Cz_sem[t_1:t_2],color=colors[tc],alpha=0.5)
plt.xlim([-0.05,0.3])
plt.xlabel('Time (sec)',fontsize=12)
plt.ylabel('\u03BCV',fontsize=12)
plt.xticks([0, 0.1, 0.2, 0.3])
#plt.yticks([-10, 0, 10])
#plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 
plt.tick_params(labelsize=12)
#plt.title('Average mod-TRF across 9 Particpants',fontsize=14)

plt.savefig(os.path.join(fig_path,'ModTRF_s_avg.svg'),format='svg')



#%% PCA average

t_cuts = [.016, .033, .066, .123, .268 ]

#32 ch average
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks[s]==ch)

Avg_Ht = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht[A_ch_picks[s],:] += A_Ht[s]

Avg_Ht = Avg_Ht / perCh

pca_sp_cuts = []
pca_expVar_cuts = []
pca_coeff_cuts = []
pca_expVar2s = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # correct polarity by looking at channel Cz
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    pca_sp_cuts.append(pca_sp)
    pca_expVar_cuts.append(pca_expVar)
    pca_coeff_cuts.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])

    


fig = plt.figure()
fig.set_size_inches(8,3)
labels = ['comp1', 'comp2']
vmin = pca_coeff_cuts[-1][0,:].mean() - 2 * pca_coeff_cuts[-1][0,:].std()
vmax = pca_coeff_cuts[-1][0,:].mean() + 2 * pca_coeff_cuts[-1][0,:].std()
for t_c in range(len(t_cuts)):
    ax = plt.subplot(1,len(t_cuts),t_c+1)
    plt.title(str(np.round(pca_expVar_cuts[t_c][0]*100)) + '%',{'horizontalalignment':'right'})
    #mne.viz.plot_topomap(pca_coeff_cuts[t_c][0,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][0,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vlim=(vmin,vmax),axes=ax)
    # plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    # plt.title('ExpVar ' + str(np.round(pca_expVar_cuts[t_c][1]*100)) + '%')
    # mne.viz.plot_topomap(pca_coeff_cuts[t_c][1,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    

plt.savefig(os.path.join(fig_path,'ModTRF_passive_sources.svg'),format='svg')




#%% Avg Hf
fs = 4096.0
t_0 = np.where(t>=0)[0][0]
t_1 = np.where(t>=0.5)[0][0]

Cz_hf = np.fft.fft(Cz_sub[:,t_0:t_1],axis=1,n=np.round(0.5*fs))  / (np.round(0.5*fs))
f = np.fft.fftfreq(Cz_hf.shape[1],d=1/fs)
phase = np.unwrap(np.angle(Cz_hf),axis=1)


Cz_hf = np.abs(Cz_hf[:,f>=0])
phase = phase[:,f>=0]
f = f[f>=0]

Cz_hf_sem = Cz_hf.std(axis=0) / np.sqrt(Cz_hf.shape[0])

phase_mn = phase.mean(axis=0)
phase_sem = phase.std(axis=0) / np.sqrt(phase.shape[0])

fig,ax = plt.subplots(2,1,sharex=True)

#ax[0].plot(t,Cz_mean,color='k')
#ax[0].fill_between(t,Cz_mean-Cz_sem,Cz_mean+Cz_sem,color='k',alpha=0.5)
#ax[0].set_xlim([-0.05,0.5])

ax[0].plot(f,Cz_hf.mean(axis=0),color='k')
ax[0].fill_between(f,Cz_hf.mean(axis=0)-Cz_hf_sem, Cz_hf.mean(axis=0) + Cz_hf_sem,alpha=0.5,color='k')
ax2 = ax[1]
ax2.plot(f,phase_mn,color='grey')
ax2.fill_between(f,phase_mn-phase_sem,phase_mn+phase_sem,color='grey',alpha=0.5)
ax[0].set_xlim([0,75])
ax[0].set_ylabel('\u03BCV')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Phase (Radians)')

plt.savefig(os.path.join(fig_path,'ModTRF_avg_f.svg'),format='svg')
plt.savefig(os.path.join(fig_path,'ModTRF_avg_f.png'),format='png')

# Just magnitude

fig,ax = plt.subplots()
ax.plot(f,Cz_hf.mean(axis=0),color='black')
ax.fill_between(f,Cz_hf.mean(axis=0)-Cz_hf_sem, Cz_hf.mean(axis=0) + Cz_hf_sem,alpha=0.5,color='black')
ax.set_ylabel('\u03BCV',fontsize=16)
ax.set_xlabel('Modulation Frequency (Hz)',fontsize=16)
ax.set_xlim([0,75])
#ax.set_yticks([0, 2e-4, 4e-4])
ax.set_xticks([0,10,20,30,40,50,60,70])
ax.tick_params(labelsize=14)
#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 
#ax.set_title('Average mod-TRF across 9 participants',fontsize=14)


plt.savefig(os.path.join(fig_path,'ModTRF_avg_fmag.svg'),format='svg')

# Just Phase
fig,ax = plt.subplots()
ax.plot(f,phase_mn,color='grey')
ax.fill_between(f,phase_mn-phase_sem,phase_mn+phase_sem,color='grey',alpha=0.5)
ax.set_ylabel('Phase (Radians)',fontsize=18)
ax.set_xlabel('Modulation Frequency (Hz)',fontsize=18)
ax.set_xlim([0,75])
ax.tick_params(labelsize=16)
#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 
ax.set_xticks([0,10,20,30,40,50,60,70])
ax.set_yticks([-2e1, -1e1, 0])
fig.set_size_inches(9,6)
plt.savefig(os.path.join(fig_path,'ModTRF_avg_phase.svg'),format='svg')

#compute group delay

f_1 = np.where(f>=10)[0][0]
f_2 = np.where(f>=20)[0][0]

coeffs = np.zeros(phase.shape[0])
coeffs1 = np.zeros(phase.shape[0])
for pp in range(phase.shape[0]):
    phz = phase[pp,:]
    coeffs[pp] = np.polyfit(f[f_1:f_2],phase[pp,f_1:f_2],deg=1)[0]
    coeffs1[pp] = np.polyfit(f[f_1:f_2],phase[pp,f_1:f_2],deg=1)[1]
gd1 = -coeffs / (2*np.pi)

gd1_mean = gd1.mean()
gd1_sem = gd1.std() / np.sqrt(gd1.size)

plt.figure()
plt.plot(f,phase.T)

plt.figure()
plt.plot(f,phase[0,:].T)
plt.plot(f,coeffs[0]*f+coeffs1[0],'k')
plt.plot(f,phase[8,:].T)


#from 40-60
f_3 = np.where(f>=30)[0][0]
f_4 = np.where(f>=70)[0][0]

coeffs2 = np.zeros(phase.shape[0])
coeffs3 = np.zeros(phase.shape[0])
for pp in range(phase.shape[0]):
    phz = phase[pp,:]
    coeffs2[pp] = np.polyfit(f[f_3:f_4],phase[pp,f_3:f_4],deg=1)[0]
    coeffs3[pp] = np.polyfit(f[f_3:f_4],phase[pp,f_3:f_4],deg=1)[1]
gd2 = -coeffs2 / (2*np.pi)

gd2_mean = gd2.mean()
gd2_sem = gd2.std() / np.sqrt(gd2.size)


plt.figure()
plt.plot(f,phase.T)

plt.figure()
plt.plot(f,phase[2,:].T)
plt.plot(f,coeffs2[2]*f+coeffs3[2],'k')

plt.plot(f,phase[5,:].T)









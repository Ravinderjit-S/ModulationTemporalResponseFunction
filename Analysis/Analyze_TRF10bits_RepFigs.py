#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:27:57 2021

@author: ravinderjit

Make Figures for mod-TRF exploring repeatability, frequency response, looking at sources etc.
"""
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
import matplotlib.pyplot as plt
import scipy.io as sio



fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')


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
Subjects = ['S207', 'S211', 'S228','S236','S238'] 

A_Tot_trials_old = []
A_Ht_old = []
A_Htnf_old = []
A_info_obj_old = []
A_ch_picks_old = []

A_Ht_old_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    print('Loading ' + subject)
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

fig = plt.figure()
fig.set_size_inches(7,4.5)
plt.plot(t*1000,cz*1e6, color='k',linewidth = 2)
plt.xlim([-100,500])
#plt.xticks([7.3,29, 47, 94, 201, 500, 1000],labels=['7.3','29','47','94','201','500','1000'])
plt.xlabel('Time (msec)', fontsize=12)
plt.ylabel('\u03BCV',fontsize=12)
plt.tick_params(axis='both', labelsize=12)
#plt.title('mod-TRF Ch. Cz',fontsize=14)
#plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#plt.xscale('log')

plt.savefig(os.path.join(fig_path,'ModTRF_ex.svg'),format='svg')


#%% Manual Peak Cutoff Reading CZ and FP
    
cuts_tms = []
#S207
cuts_tms.append([.0185, .036, .067, .136 ,.266])

#S211
cuts_tms.append([.016, .037, .063, .123, .300])

#S228
cuts_tms.append([.016, .043, .069, .125, .238])

#S236
cuts_tms.append([.014, .031, .066, .124, .249])

#238
cuts_tms.append([.0155, .045, .062, .124, .287])

#S250
cuts_tms.append([.016, .049, .123, .220, .334])
      

#%%

# fig = plt.figure()
# ax = [None] *5
# ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
# ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
# ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
# ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

fs = 4096
fig,ax = plt.subplots(3,2)
fig.set_size_inches(10,10)

t_0 = np.where(t_epochs>=0)[0][0]
colors = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7']

source_numerals = ['I', 'II', 'III', 'IV', 'V']
for sub in np.arange(len(Subjects[:3])):
    #Plot Cz
    
    if sub <2:
        ax[sub,0].axes.xaxis.set_visible(False)
        ax[sub,1].axes.xaxis.set_visible(False)
        
    Ht_mean_old = A_Ht_old_epochs[sub].mean(axis=1) 
    Ht_sem_old = A_Ht_old_epochs[sub].std(axis=1) / np.sqrt(A_Ht_old_epochs[sub].shape[1])

    cz_mean_v1 = Ht_mean_old[-1,:] - Ht_mean_old[-1,t_0]
    cz_mean_v1 = cz_mean_v1 * 1e6
    
    Ht_sem_old = Ht_sem_old*1e6
    
    ax[sub,0].plot(t_epochs, cz_mean_v1,color='k',label = '1st Visit')
    #ax[sub,0].fill_between(t_epochs,cz_mean_v1 -Ht_sem_old[-1,:],cz_mean_v1 + Ht_sem_old[-1,:], color='k',alpha=0.5)
    
    t_cuts = cuts_tms[sub]
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        #ax[sub,0].plot(t_epochs[t_1:t_2], cz_mean_v1[t_1:t_2],color=colors[t_c])
        ax[sub,0].fill_between(t_epochs[t_1:t_2],cz_mean_v1[t_1:t_2] -Ht_sem_old[-1,t_1:t_2],cz_mean_v1[t_1:t_2] + Ht_sem_old[-1,t_1:t_2], color=colors[t_c],alpha=0.8)
    
    
    ax[sub,0].set_xlim([-0.05,0.5])
    ax[sub,0].set_title('P' + str(sub+1),fontweight='bold',fontsize=14)
    ax[sub,0].set_xticks([0,0.1,0.2,0.3,0.4])
    #ax[sub,0].set_yticks([-.002,0,.002])
    #ax[sub,0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    
    t_05 = np.where(t_epochs>=t_cuts[-1])[0][0]
    cz_mean_v1 = cz_mean_v1[t_0:t_05]
    cz_mean_v1 = cz_mean_v1 - cz_mean_v1.mean()
    Ht_freq_v2 = np.fft.fft(cz_mean_v1[:], int(np.round(0.3*fs))) / (np.round(0.3*fs))
    f = np.fft.fftfreq(Ht_freq_v2.size,d=1/fs)
    
    Ht_freq_v2 = Ht_freq_v2[f>=0]
    f = f[f>=0]

    lines = []
    l1, = ax[sub,1].plot(f,np.abs(Ht_freq_v2),color='k',label='Whole',linewidth=3)
    lines.append(l1)

    
    
    
    ax[sub,1].set_xlim([0,75])
    
    t_cuts = cuts_tms[sub]
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        t_ep = t_epochs[t_1:t_2]
        
        Ht_freq = np.abs(np.fft.fft(A_Ht_old_epochs[sub][-1,:,t_1:t_2] - A_Ht_old_epochs[sub][-1,:,t_1:t_2].mean(axis=-1)[:,np.newaxis],n=int(np.round(0.3*fs)))) / (np.round(0.3*fs))
        Ht_freq_mean = Ht_freq.mean(axis=0) * 1e6
        Ht_freq_sem = (Ht_freq.std(axis=0) / np.sqrt(Ht_freq.shape[0]))*1e6
                   
        f_t = np.fft.fftfreq(Ht_freq_mean.size,d=1/fs)
        
        Ht_freq_mean = Ht_freq_mean[f_t >= 0]
        Ht_freq_sem = Ht_freq_sem[f_t>=0]
        f_t = f_t[f_t>=0]


        l1, = ax[sub,1].plot(f_t,Ht_freq_mean,color=colors[t_c], label = source_numerals[t_c])
        lines.append(l1)
        ax[sub,1].fill_between(f_t,Ht_freq_mean-Ht_freq_sem,Ht_freq_mean+Ht_freq_sem,color=colors[t_c],alpha=0.5)

    if sub == 0:
        ax[0,1].legend(lines,[l.get_label() for l in lines],fontsize=12)
    
    
for a_ in ax.flatten():
    a_.tick_params(axis='both', labelsize=12)
ind_set = 2
    
ax[ind_set,0].set_xlim([-0.05,0.4])
ax[ind_set,0].set_xlabel('Time (sec)',fontsize=14)
ax[ind_set,0].set_ylabel('\u03BCV',fontsize=14)


ax[ind_set,1].set_xticks([0,25, 50, 75])
ax[ind_set,1].set_xlabel('Frequency (Hz)',fontsize=14)
ax[ind_set,1].set_ylabel('\u03BCV',fontsize=14)

ax[0,0].set_yticks([-15,0,15,30])
ax[1,0].set_yticks([-5,0,5])
ax[2,0].set_yticks([0,15,30])

ax[0,1].set_yticks([-0,4,8])
ax[1,1].set_yticks([0,2])
ax[2,1].set_yticks([0,2,4])


plt.tick_params(labelsize=12)

plt.savefig(os.path.join(fig_path,'ModTRF_tf_source.svg'),format='svg')
plt.savefig(os.path.join(fig_path,'ModTRF_tf_source.png'),format='png')


#%% Just plot time domain

fs = 4096
fig,ax = plt.subplots(2,3,sharex=True)
fig.set_size_inches(15,10)

ax = np.reshape(ax,6)
t_0 = np.where(t_epochs>=0)[0][0]

for sub in np.arange(len(Subjects)):
    #Plot Cz
    
    if sub !=3:
        ax[sub].axes.xaxis.set_visible(False)
        #ax[sub].axes.yaxis.set_visible(False)
        
    Ht_mean_old = A_Ht_old_epochs[sub].mean(axis=1) *1e6
    Ht_mean = A_Ht_epochs[sub].mean(axis=1) * 1e6
    
    Ht_sem_old = A_Ht_old_epochs[sub].std(axis=1) / np.sqrt(A_Ht_old_epochs[sub].shape[1]) *1e6
    Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1]) *1e6
    
    cz_mean_v1 = Ht_mean_old[-1,:] - Ht_mean_old[-1,t_0]
    
    ax[sub].plot(t_epochs, cz_mean_v1,color='k',label = '1st Visit')
    ax[sub].fill_between(t_epochs,cz_mean_v1 -Ht_sem_old[-1,:],cz_mean_v1 + Ht_sem_old[-1,:], color='k',alpha=0.5)
    
    cz_mean_v2 = Ht_mean[-1,:] - Ht_mean[-1,t_0] #Look at cz. Make time 0 start at 0
    
    ax[sub].plot(t_epochs, cz_mean_v2,color='grey', label='2nd Visit')
    ax[sub].fill_between(t_epochs,cz_mean_v2 -Ht_sem[-1,:],cz_mean_v2 + Ht_sem[-1,:], color='grey',alpha=0.4)
    

    #ax[sub].set_xscale('log')
    ax[sub].set_title('P' + str(sub+1),fontweight='bold',fontsize=14)
    
for a_ in ax:
    a_.tick_params(axis='both', labelsize=14)

ax[3].set_xlim([-0.05,0.5])
ax[3].set_xlabel('Time (s)',fontsize=14)
ax[3].set_ylabel('\u03BCV',fontsize=14)
ax[3].set_xticks([0,0.1,0.2,0.3,0.4])
#ax[3].set_yticks([-.002,0,.002,.004])
ax[3].tick_params(labelsize=14)
ax[0].legend(fontsize=14)

ax[0].set_yticks([-15,0,15,30])
ax[1].set_yticks([-5,0,5,10])
ax[2].set_yticks([-10,0,15,30])
ax[3].set_yticks([-15,0,15,30])
ax[4].set_yticks([-10,0,10,20])
ax[5].set_yticks([-15,0,15,30])

plt.savefig(os.path.join(fig_path,'ModTRF_rep_t.svg'),format='svg')
plt.savefig(os.path.join(fig_path,'ModTRF_rep_t.png'),format='png')


#%% Look at epochs with t cuts and in freq domain

# colors = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

# fs = 4096
# fig = plt.figure()
# ax = [None] *5
# ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
# ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
# ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
# ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

# fig_f = plt.figure()
# axf = [None] *5
# axf[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# axf[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
# axf[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
# axf[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
# axf[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

# for sub in np.arange(len(Subjects)):
#     #Plot Cz
    
#     if sub > 0:
#         ax[sub].axes.yaxis.set_visible(False)
#         axf[sub].axes.yaxis.set_visible(False)
        
#     Ht_mean = A_Ht_epochs[sub].mean(axis=1)
#     Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1])
    

#     t_cuts = cuts_tms[sub]
#     for t_c in range(len(t_cuts)):
#         if t_c ==0:
#             t_1 = np.where(t_epochs>=0)[0][0]
#         else:
#             t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
#         t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
#         t_ep = t_epochs[t_1:t_2]
#         Ht_mean_tc = Ht_mean[-1,t_1:t_2] #- Ht_mean[-1,t_1:t_2].mean()
        
#         Ht_freq = np.abs(np.fft.fft(A_Ht_epochs[sub][-1,:,t_1:t_2] - A_Ht_epochs[sub][-1,:,t_1:t_2].mean(axis=-1)[:,np.newaxis]))
#         Ht_freq_mean = Ht_freq.mean(axis=0)
#         Ht_freq_sem = Ht_freq.std(axis=0) / np.sqrt(Ht_freq.shape[0])     
                   
#         f = np.fft.fftfreq(Ht_freq_mean.size,d=1/fs)

#         ax[sub].plot(t_ep, Ht_mean_tc, color=colors[t_c])
#         ax[sub].fill_between(t_ep,Ht_mean_tc -Ht_sem[-1,t_1:t_2],Ht_mean_tc + Ht_sem[-1,t_1:t_2], color=colors[t_c],alpha=0.5)
#         ax[sub].set_title('Subject ' + str(sub+1))
#         ax[sub].set_xticks([0,0.1,0.2,0.3,0.4])
        
#         axf[sub].plot(f,Ht_freq_mean,color=colors[t_c])
#         axf[sub].fill_between(f,Ht_freq_mean-Ht_freq_sem,Ht_freq_mean+Ht_freq_sem,color=colors[t_c],alpha=0.5)
#         axf[sub].set_title('Subject ' + str(sub+1))
#         axf[sub].set_xlim([0,75])
#         axf[sub].set_xticks([10,25,50])
        


# ax[0].set_xlabel('Time (sec)')
# ax[0].set_ylabel('Amplitude')
# ax[0].set_yticks([-.002,0,.002,.004])
# ax[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# #ax[0].legend(fontsize=9)
# #fig.suptitle('Ch. Cz',fontweight='bold')

# axf[0].set_xlabel('Frequency (Hz)')
# axf[0].set_ylabel('Magnitude')
# axf[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))






# EEG Processing of Mod-TRF
This folder contains the code to analyze the EEG data recorded from a em-seq amplitude modulation.
- mseqHelper.py contains functions to do the cross correlation with an mseq
- AnalyzeEEG_AMmseq_10bits.py, AnalyzeEEG_AMmseq_10bits_active.py, AnalyzeEEG_AMmseq_activeHarder.py, and AnalyzeEEG_AMmseq_bits4.py loads EEG data, preprocesses EEG data, cross correlates EEG with the em-seq to obtain a mod-TRF, and saves that result. 
    - Each file corresponds to a different condition that was recorded
- Analyze_TRF_bitsFig.py analyzes the 4 different bit values used to obtain a mod-TRF
- Analyze_TRF10bits_RepFigs.py includes plotting two mod-TRFs measured months apart for particpants. It also includes a source analysis in the frequency domain for the individual sources in the mod-TRF.
- Analyze_TRF10bits_avgFig.py plots the average mod-TRF across participants in time domain and frequency domain, and the topomaps from a source analysis using PCA.
- Analyze_TRF10bits_activeFigs.py evaluates the effects of attention on the mod-TRF    
function [stim] = AM_mseq(mseqAM,Point_len)
%m = bits of mseq
%upperF = highest frequency to characterize with mseq
%fs = sampling rate

mseqAM = (mseqAM+1)/2; %make mseq bounce b/t 0 & 1
w = dpss(Point_len,1,1);
w = w-w(1); w = w/max(w); %dpss ramp so energy in modulation doesn't spread out far beyond upperF

mseqAM = conv(mseqAM,w,'same');
mseqAM = mseqAM / max(mseqAM);

nn = randn(1,length(mseqAM));
stim = nn.*mseqAM;
stim = [stim;stim];

end




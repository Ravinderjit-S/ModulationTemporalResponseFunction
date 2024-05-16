clear

fname = 'mseqEEG_500_bits11';
load([fname '.mat'])
%load('/media/ravinderjit/Storage2/EEGdata/mseqEEG_80.mat')

fs= 48828;

resample_fs = 4096;
%resample_fs = 16384;


m2 = resample(mseqEEG,resample_fs,fs);
m2(m2<0) = -1;
m2(m2>0) = 1;

t = 0:1/fs:length(mseqEEG)/fs -1/fs;
t2 = 0:1/resample_fs:length(m2)/resample_fs-1/resample_fs;
figure,plot(t,mseqEEG,'b',t2,m2,'r')
ylim([-1.1,1.1])

mseqEEG_4096 = m2;
save([fname '_'  num2str(resample_fs)  '.mat'],'mseqEEG_4096','Point_len')



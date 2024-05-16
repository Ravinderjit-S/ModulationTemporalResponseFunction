function [mseqEEG,Point_len] = EEGmseq(m,upperF,fs,reps)

Point_len = floor(fs/upperF);
mseq = mls(m,0);

mseqEEG = [];
for i =1:length(mseq)
    mseqEEG = [mseqEEG ones(1,Point_len)*mseq(i)];
end

mseqEEG = repmat(mseqEEG,1,reps);

end
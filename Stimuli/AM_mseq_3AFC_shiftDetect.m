function [stim] = AM_mseq_3AFC_shiftDetect(mseqAM,Point_len,shift)

A1 = AM_mseq(mseqAM,Point_len);
A2 = AM_mseq(mseqAM,Point_len);
B = AM_mseq(circshift(mseqAM,shift),Point_len);

stim = {A1,A2,B};

end
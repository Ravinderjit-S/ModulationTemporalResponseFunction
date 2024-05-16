clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');

fig_num=99;
USB_ch=1;
FS_tag = 3;
[f1RZ,PS.RP,FS]=load_play_circuit(FS_tag,fig_num,USB_ch);


%% Stim & Experimental parameters

L=75; %dB SPL
ntrials = 100;
nconds =1;
jitlist = rand(1,ntrials*nconds)*0.1;
risetime = 1/150; 
TypePhones = 'earphones';
fs =48828.125;
passive =0;

mseq = load('mseqEEG_150_bits10.mat');

repeats = [3, 3, 2, 1]; %will repeat 3 times 50%, 2 times 25%, and 1 time, 25 %

respList = []
correctList = []


%% Startup parameters
FsamptTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =0;

feedbackDuration =0.2;

% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);


for i =1:ntrials*nconds
    if mod(i,20) == 0 % optional break every 20 trials
      % Turns EEG Saving off ('Pause on')
        invoke(PS.RP, 'SetTagVal', 'trgname', 254);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        
        fprintf(1,'Break %d/%d \n',i/20,4)
        info = sprintf('Break %d/%d! You are about to start trial %d out of %d',i/20,4,i,nconds*ntrials);
        info2 = sprintf('Press any button twice to resume');
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        Screen('DrawText',PS.window,info2,textlocH,textlocV+100,PS.white);
        Screen('Flip',PS.window);
        if buttonBox  %Subject pushes button twice
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        Screen('Flip',PS.window);
        % Turns EEG Saving on ('Pause off')
        invoke(PS.RP, 'SetTagVal', 'trgname',253);
        invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
        invoke(PS.RP, 'SoftTrg', 6);
        pause(2.0);
    end
  
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    repeat_i = repeats(randi(length(repeats)));
    for j = 1:repeat_i
        stim_dur = length(mseq.mseqEEG)/fs;
        tic();
        stim = AM_mseq(mseq_j.mseqEEG,mseq_j.Point_len);
        genStimTime = toc();
        PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', order(j), TypePhones);
        WaitSecs(stim_dur - genStimTime + 0.5 + rand()*0.1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    correct = repeat_i;
    WaitSecs(0.3); %wait until show dot for response 
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correct);
    
    respList = [respList, resp]; %#ok<AGROW>
    correctList = [correctList,correct]; %#ok<AGROW>
    
end
save(strcat(subj, '_AMmseqActive'), 'respList','correctList');

% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(f1RZ,PS.RP);
fprintf(1,'\n Done with data collection!\n');




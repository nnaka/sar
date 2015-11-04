function [rise,fall,group_num]=process_sync(x)
% function [rise,fall,group_num]=process_sync(x)
%
% identify starts and stops of pulses and pulse groups

% adjustable parameters
samples_per_sec=44100;
height_thresh=200;
t_min_pulse=15e-3; % minimum duration of a pulse
t_max_pulse=25e-3; % maximum duration of a pulse
t_min_break=0.5; % minimum duration of gap between pulse groups
n_min_pulses_per_group=25; % minimum number of pulses per group
n_discard_first=3; % drop this many pulses at the beginning of a group
n_discard_last=3; % drop this many pulses at the end of a group

cleanup=true; % save memory
be_verbose=true;

pulsewidth_thresh=floor(t_min_pulse*samples_per_sec);
breakwidth_thresh=floor(t_min_break*samples_per_sec);

% find when signal has been above a threshold for some duration
x1=x>=height_thresh;
x2=conv(double(x1),ones(pulsewidth_thresh,1),'same');
if cleanup, clear x1, end
x3=x2>=pulsewidth_thresh;
if cleanup, clear x2, end
x4=[0;diff(x3)];
if cleanup, clear x3, end

% initial rise and fall estimates 
rise0=find(x4==1);
fall0=find(x4==-1);
if cleanup, clear x4, end
assert(all(fall0>rise0))

% zero crossings of original signal
x5=x>0;
%x6=[sign(x(1));diff(x5)]; % assumes first sample is a rise or fall - good?
%x6=[double(x(1));diff(x5)]; % assumes first sample is a rise if high
%x6=[0;diff(x5)]; % assumes first sample is never a rise or fall

% this will assume if the first sample is high, it's a rise, and if the
% last sample is high, then the following sample is a fall
x6=diff([false;x5;false]); 
if cleanup, clear x5, end

% candidate rise and fall times (zero crossings)
rise_candidates=find(x6==1);
fall_candidates=find(x6==-1);
if cleanup, clear x6, end
assert(all(fall_candidates>rise_candidates))

% for each rise and fall estimate, find nearest candidate
% (this crashes if there's only one rise or fall candidate)
rise=interp1(rise_candidates,rise_candidates,rise0,'nearest','extrap');
fall=interp1(fall_candidates,fall_candidates,fall0,'nearest','extrap');

assert(~any(isnan(rise)|isnan(fall)))

% this can happen with small positive signals because rise0 is based on
% (non-zero) threshold crossings, and rise_candiates is based on zero
% crossings
ind_discard=(rise>rise0)|(fall<fall0);
rise(ind_discard)=[];
fall(ind_discard)=[];

assert(all(fall>rise))

if be_verbose,display(['found ',num2str(length(rise)),' raw pulses!']),end

% filter on pulsewidth
delta=fall-rise;
ind_discard=(delta<t_min_pulse*samples_per_sec)|(delta>t_max_pulse*samples_per_sec);
rise(ind_discard)=[];
fall(ind_discard)=[];
if be_verbose,display(['discarded ',num2str(nnz(ind_discard)),' pulses based on pulsewidth']),end

% find which group each pulse is in
if length(rise)>1
    is_first_of_group=[true;diff(rise)>=breakwidth_thresh];
else
    is_first_of_group=true(length(rise),1);
end

% filter on number of pulses per group here
first_pulse_ind=find(is_first_of_group);
last_pulse_ind=[first_pulse_ind(2:end)-1;length(is_first_of_group)];
pulses_per_group=1+last_pulse_ind-first_pulse_ind;
ind_discard_group=(pulses_per_group<n_min_pulses_per_group);
ind_discard=false(size(rise));
for k=1:length(pulses_per_group)
    if ind_discard_group(k)
        ind_discard(first_pulse_ind(k):last_pulse_ind(k))=true;
    end
end
rise(ind_discard)=[];
fall(ind_discard)=[];
is_first_of_group(ind_discard)=[];
if be_verbose,display(['discarded ',num2str(nnz(ind_discard)),' pulses based on pulses per group']),end

% find which group each pulse belongs to
group_num=cumsum(is_first_of_group);

% throw away first and last n pulses in each group
temp=ones(n_discard_first+n_discard_last,1);
ind_discard=logical(conv(double(is_first_of_group),temp));
ind_discard(1:n_discard_last)=[]; % first n_discard_last pts
ind_discard((end-n_discard_first+2):end)=[]; % last n_discard_first-1 pts
ind_discard=logical(ind_discard);

rise(ind_discard)=[];
fall(ind_discard)=[];
group_num(ind_discard)=[];
if be_verbose,display(['discarded ',num2str(nnz(ind_discard)),' pulses at beginnings and ends of groups']),end

if be_verbose,display(['final answer: ',num2str(length(rise)),' pulses in ',num2str(group_num(end)),' groups!']),end

% % plot pulses with groups
% tempx=[
%     rise';
%     rise';
%     fall';
%     fall';
%     ];
% tempy=[
%     zeros(1,length(rise));
%     group_num';
%     group_num';
%     zeros(1,length(rise));
%     ];
% figure,plot(tempx(:),tempy(:))
return

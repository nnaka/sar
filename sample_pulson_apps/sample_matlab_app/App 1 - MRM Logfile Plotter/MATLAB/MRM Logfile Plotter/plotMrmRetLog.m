% plotMrmRetLog.m
% This script prompts the user for a MRM-RET logfile, reads, parses, and
% produces a "waterfall plot" of the motion filtered scans and detection lists 
% in the logfile
clear all; close all; clc

%% Query user for logfile
%dnm = '.'; fnm = 'MRM_002.csv';
[fnm,dnm] = uigetfile('*.csv');
fprintf('Reading logfile %s\n',fullfile(dnm,fnm));
[cfg,req,scn,det] = readMrmRetLog(fullfile(dnm,fnm));

%% Separate raw, bandpassed, and motion filtered data from scn structure
% (only motion filtered is used)

%% Pull out the raw scans (if saved)
rawscansI = find([scn.Nfilt] == 1);
rawscansV = reshape([scn(rawscansI).scn],[],length(rawscansI))';
% band-pass filtered scans
bpfscansI = find([scn.Nfilt] == 2);
bpfscansV = reshape([scn(bpfscansI).scn],[],length(bpfscansI))';
% motion filtered scans
mfscansI = find([scn.Nfilt] == 4);
mfscansV = reshape([scn(mfscansI).scn],[],length(mfscansI))';


%% Create the waterfall horizontal and vertical axes
Tbin = 32/(512*1.024);  % ns
T0 = 0; % ns
c = 0.29979;  % m/ns
Rbin = c*(Tbin*(0:size(mfscansV,2)-1) - T0)/2;  % Range Bins in meters

IDdat = [scn(mfscansI).msgID]; % msgID == scanID


%% The envelope of the motion filtered scans is a low pass of abs value
fprintf('Computing the envelope of the motion filtered data...\n');

%[b,a] = butter(6,0.4);  % only available with sig proc toolbox
% Hard code instead
b = [0.0103 0.0619 0.1547 0.2063 0.1547 0.0619 0.0103];
a = [1.0000 -1.1876 1.3052 -0.6743 0.2635 -0.0518 0.0050];
edat = max(filter(b,a,abs(mfscansV),[],2),0);

%% Plot enveloped motion filtered data as a waterfall
fprintf('Plotting motion filtered data as a waterfall plot...\n');

figure('Units','normalized','Position',[0.1 0.2 0.7 0.7],'Color','w')
imagesc(Rbin,IDdat,edat);
hold on
xlabel('R (m)')
ylabel('Scan Number')
title('Waterfall plot of motion filtered scans')
drawnow

%% Optionally plot waterfall with detection list overplot
% Note detection list has maximum of 350 points
OVERPLOT_DETECTIONS = 1;
if OVERPLOT_DETECTIONS
figure('Units','normalized','Position',[0.1 0.2 0.7 0.7],'Color','w')
imagesc(Rbin,IDdat,edat);
hold on
xlabel('R (m)')
ylabel('Scan Number')
title('Waterfall plot of motion filtered scans')

% Map the detection list mapping to scan number and distance
fprintf('Mapping detections to waterfall...\n');
for i = 1:length(det)
  IDdet = det(i).msgID;
  if det(i).Ndet > 0
    Idet = det(i).det(1,:) + 1;  % MATLAB 1,2,... versus C 0,1,...
    
    plot(Rbin(Idet),IDdet,'r.');
  end
end
title({'Waterfall plot of motion filtered scans','Overplotted with detection points'})

end

break % The rest takes a while and doesn't add much so we added a break here

% Compare the detection list magnitudes to motion filtered magnitudes.
% These should be the same except detection list saturates at 2^16-1
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
hold on
xlabel('Scan Number')
ylabel('Amplitude')
title('Detection List Magnitudes & Motion Filter Magnitudes vs. Scan Number')
for i = 1:length(det)
  IDdet = det(i).msgID;
  if det(i).Ndet > 0
    % Index of the detection
    Idet = det(i).det(1,:) + 1;  % MATLAB 1,2,... versus C 0,1,...
    % Amplitude of the detection
    Adet = det(i).det(2,:);
    
    % Map detections to data
    j = find(IDdet == IDdat);
    edet = edat(j,Idet);
    
    % Overplot detection amplitudes with motion filtered amplitudes.
    hAdet = plot(IDdet,Adet,'r.');
    hEdet = plot(IDdet,edet,'ko');
  end
end
legend([hAdet(1) hEdet(1)],'Detection List Amplitudes','Enveloped Motion Filter Amplitudes',0)

% NOTE: RET-MRM also adds a clustering such that 3 or more detections in a
% row must be present before it's displayed in the RET scan window

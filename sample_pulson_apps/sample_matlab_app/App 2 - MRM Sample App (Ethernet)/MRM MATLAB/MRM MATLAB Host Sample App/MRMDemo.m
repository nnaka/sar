% MRM Demo
% Hand-held Radar
%
% This demo illustrates direct access from MATLAB to the MR through
% the MRM API and UDP sockets.
%
% This code was initially developed by senior design students at the University
% of Alabama in Huntsville (UAHuntsville) and targeted at a  
% hand-held radar.  Please suggest improvements to:
% Please direct comments, suggestions, or improvements to:
% aeh0005<at>uah.edu.
%
% This code should work on any PC running MATLAB.  It has been tested
% against Windows7 and Mac OSX running MATLAB.  We used a UWB horn
% antenna to improve handheld directionality.
%
% INSTRUCTIONS:
% - Connect MRM using an ethernet cable.
% - Power up and test the connection using 'ping' in a cmd window.
% - Adjust the mrmIpAddr below to match that of your MRM.
% - Execute this script from the MATLAB command prompt.
% - Any the change in reflectivity relative to the first scan will be
%   displayed.
% 
% This script illustrates:
% 1. Connecting to the MRM and querying configuration using the
% MRM_GET_CONFIG_REQUEST command (see MRM API.)
% 2. Parsing the MRM_GET_CONFIG_CONFIRM structure with the big-endian
% byte swap required on INTEL processors.
% 3. Adjusting the CONFIG and setting start, stop, and integration based on
% hand-held radar preferences using the MRM_SET_CONFIG_REQUEST message.
% 4. Loop, commanding, receiving, processing, and plotting one radar scan 
% at a time.  
% 5. In the included algorithm the INITIAL SCAN is used as a template and
% subtracted from all following scans.  Each "residual" scan is rectified
% and low-passed to produce an envelope.  This envelope is thresholded
% to produce a detection list.
%
% NOTE: The algorithm demonstrated is targeted at an Android tablet
% with modest communication and processing capability.  Therefore we 
% rather than commanding the MRM to continuous scan mode (which could 
% overburden our host) our main loop iteratively requests a single scan 
% from the MRM after which we process and display before requesting
% another.
% This adds ~2ms of jitter to our scan timing but since we aren't using
% a motion filter (we use the first scan as a reference template) scan
% timing isn't so important.

clear all; close all; clc

%% Initialize Constants
mrmIpAddr = '192.168.1.102';
%mrmIpAddr = '192.168.1.100';
scanStartPs = 11000; % Adjust this to match antenna delay
C_mps = 299792458;
maxDistance_m = 4;  % MRM will quantize to closest above this number
scanStopPs = scanStartPs + (2*maxDistance_m/C_mps)*1e12; % 1e12 ps in one sec
pulseIntegrationIndex = 12; % The number of pulses per scan point
transmitGain = 0; % Tx power (0 for FCC legal)
scanIntervalTime_ms = 143; % Time between start of each scan in millisecs
scanRes_ps = 61; % 61 picoseconds between each data point (from API.) Used for plotting.
    
%% Set up the plot window
screensize = get(0,'ScreenSize');
winLeft = 50; winBottom = 50;
winWidth = screensize(3)-100;
winHeight = screensize(4)-100;
mfh = figure('Position',[winLeft winBottom winWidth winHeight]);
set(gca, 'xlimmode','manual','ylimmode','manual'); % No autoscale?

% Initialize the controls
number = uicontrol('style','text', ...
   'string','1', ...
   'fontsize',12, ...
   'position',[40,500,60,20]); 

startbutton = uicontrol('style','pushbutton',...
   'string','RESTART', ...
   'fontsize',12, ...
   'position',[40,450,60,20], ...
   'callback', 'restart=1;');

quitbutton = uicontrol('style','pushbutton',...
   'string','STOP', ...
   'fontsize',12, ...
   'position',[40,400,60,20], ...
   'callback','done=1;');
 
restart = 0; stop = 0; done = 0;


%% Open a socket for communicating with the MRM
sckt = sckt_mgr('get');
if isempty(sckt)
  sckt_mgr('open');
end


%% Get the configuration of the MRM
get_cfg_rqst(mrmIpAddr,1)
[msg,msgType,msgID,mrmIpAddr] = read_pckt;
if isempty(msgID)
  error('Unable to communicate with the MRM')
end
[CFG,msgType,msgID] = parse_msg(msg);


%% Update the config structure & send to the MRM
% The MRM API specifies the variable types
CFG.scanStartPs = uint32(scanStartPs);
CFG.scanStopPs = uint32(scanStopPs);
CFG.pulseIntegrationIndex = uint16(pulseIntegrationIndex); % The higher this is the higher the snr but longer it takes to scan
CFG.transmitGain = uint8(transmitGain);
set_cfg_rqst(mrmIpAddr,2,CFG);

%% Read the confirm from the MRM
[msg,msgType,msgID,mrmIpAddr] = read_pckt;
if ~strcmp(msgType,'1101') % MRM_SET_CONFIG_CONFIRM
    error(fprintf('Invalid message type: %s, should have been 1101',msgType));
end


%% Command radar to scan designated number of scans (-1 continuous)
scanCount = 1;
CTL.scanCount = uint16(scanCount); % 2^16-1 for continuous
CTL.reserved = uint16(0); % Aligns to word
CTL.scanIntervalTime = uint32(scanIntervalTime_ms*1000); % Microsecs between scan starts

loopI = 0;  % This is the loop and scan index
while (~done)
  tStart = tic;
  loopI = loopI + 1;

  % Implement restart button behavior
  if restart == 1
    loopI = 1;
    restart = 0;
  end

  % Allows you to setup before collecting the first reference scan
  % Point the radar into open space.
  if loopI == 1
    pause(1)
  end
  
  %% Request a scan and read it back
  % Request a scan
  ctl_rqst(mrmIpAddr,msgID,CTL) 
  % Read the confirm
  [msg,msgType,msgID,mrmIpAddr] = read_pckt;
  % Read the first scan msg.  Analyze the hdr for how many follow
  [msg,msgType,msgID,mrmIpAddr] = read_pckt;
  [scanInfo,msgType,msgID] = parse_msg(msg);
  scanRaw = double(scanInfo.scanData);  % Save the good stuff. Append to this later
  % Loop, reading the entire waveform scan into scanDataSaved    
  for j = 1:scanInfo.numberMessages-1
      [msg,msgType,msgID,mrmIpAddr] = read_pckt;
      [scanInfo,msgType,msgID] = parse_msg(msg);
      scanRaw = [scanRaw, double(scanInfo.scanData)];
  end

  %% Process the scan depending on the loop iteration
  %
  % In iteration 1 save the scan as a reference template
  % Note: the first scan is lousy for some reason, so changed to scan2
  if loopI == 2;
    %maxMag = max(abs(scanRaw));
    scanScaleFactor = 100.0/max(abs(scanRaw));
    scanTemplate = scanScaleFactor*scanRaw;
    distanceAxis_m = ([0:length(scanRaw)-1]*scanRes_ps/1e12)*C_mps/2;  % scanIndex*(61ps/step)/(ps/sec)*(meters/sec)/2 (round trip)
    hold off;
    plot(distanceAxis_m,scanTemplate,'Color',[0.5 0.5 0.5]);
    axis tight;
    xlabel('Distance (m)');
    ylabel('Signal Strength');
  end
  
  % On iteration 2 compute and plot the delta between raw and template
  % Low-pass filter the absolute value to estimate an envelope
  if loopI == 3
    scanLen = length(scanRaw);
    scanDelta = abs(scanScaleFactor*scanRaw - scanTemplate);
    scanEnvelope = movingAvg(scanDelta);
    minThreshold = 3*[scanEnvelope + std(scanDelta)];
    scanThreshold = (100./[1:scanLen].^0.6) + minThreshold; % add 1/r^alpha. adjust alpha until it looks right.
    hold on
    plot(distanceAxis_m,scanThreshold,'r--');
    hEnv = plot(distanceAxis_m,scanEnvelope,'b');
    hDetList = plot(distanceAxis_m(1:scanLen),zeros(1,scanLen),'r.');  % Do this to set up detection list update
    legend('Raw Scan','Threshold','Enveloped Scan','Detection List');
  end
  
  % From now on compute and plot delta scan and detection list
  if loopI >= 4
    scanDelta = abs(scanScaleFactor*scanRaw - scanTemplate);
    scanEnvelope = fir_lpf_ord5(scanDelta);
    detectionI = find(scanEnvelope > scanThreshold);
    detectionV = scanEnvelope(detectionI);

    set(hEnv,'YData',scanEnvelope);
    set(hDetList,'XData',distanceAxis_m(detectionI),'YData',detectionV);

    tElapsed = toc(tStart)*1000;
    set(number,'string',sprintf('%3.1fms',tElapsed));

    if length(detectionI) >= 3
      distance1 = distanceAxis_m(detectionI(1));
      sigStr1 = sum(scanEnvelope(detectionI(1:3)));
      fprintf('Distance: %3.2f, Magnitude: %3.2f\n',distance1, sigStr1);
    end
    
  end    
  
  drawnow;
  
end
fprintf('Quit\n');

% MIT IAP 2013: Find a Needle in a Haystack
%
% Run this script to initiate monostatic data collect at different spatial
% locations.
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

clear all;
close all;
clc;
instrreset;

addpath('support');

fprintf('======================================================\n')
fprintf('MIT IAP 2013: Find a Needle in a Haystack\n')
fprintf('Monostatic Data Collection Script\n');
fprintf('======================================================\n')

%% Initialize Constants

INCHES2METERS = 0.3048/12;
numXScans=40;
numZScans=20;
numScanLocs=numXScans*numZScans;

xTmp_m = INCHES2METERS*((0:(numXScans-1))-(numXScans-1)/2);
zTmp_m = INCHES2METERS*0.85*(0:(numZScans-1));

xLoc_m = repmat(xTmp_m,1,numZScans);
yLoc_m = zeros(1,numScanLocs);
zLoc_m = repmat(zTmp_m,numXScans,1);
zLoc_m = zLoc_m(:)';

% User selectable
%scanStartPs = 17099;        % Adjust this to match antenna delay
scanStartPs = 16810;
maxDistance_m = 5;         % P410 will quantize to closest value above this number
pulseIntegrationIndex = 15; % The number of pulses per scan point (2^n)
transmitGain = 0;           % Tx power (0 for FCC legal)
scanIntervalTime_ms = 1;    % Time between start of each scan in millisecs

% Derived parameters
C_mps = 299792458;
scanStopPs = scanStartPs + (2*maxDistance_m/C_mps)*1e12; % 1e12 ps in one sec
codeChannel = 0;            % PN code
antennaMode = 2;            % Tx: A, Rx: B
scanStepBins = 32;          
scanResPs = scanStepBins*1.907; % 61 ps sampling resolution
scanCount = 1;              % number of scans per spatial location (2^16-1 for continuous)

%% Detect Active Radios

activeSerialList = detectActiveRadios();

% Use first active radio for monostatic collection
if ~isempty(activeSerialList)
    activeSerial = activeSerialList{1};
else
	error('Unable to find active P410 radio')    
end

%% Connect to Desired Radio

% Open socket
s = serial(activeSerial,'InputBufferSize',15000,'Timeout',0.01);
fopen(s);

% Query for configuration
getConfig(s,1);

% Read packet
[msg, msgType, msgID] = readPacket(s);
if isempty(msgID)
  error('Unable to communicate with P410 radio')
end

[CFG,msgType,msgID] = parseMessage(msg);

fprintf(['Connected to node ' num2str(CFG.nodeId) ' on ' activeSerial '...\n']);

%% Set Monostatic Operation Mode

% Set operation mode
opMode = 1; % Monostatic
setOpMode(s,1,opMode);

% Read confirmation
[msg, msgType, msgID] = readPacket(s);
if ~strcmp(msgType,'F103') && ~typecast(msg(end-3:end),'uint32')
    error(fprintf('Invalid message type: %s, should have been F103',msgType));
end

fprintf(['Successfully set node ' num2str(CFG.nodeId) ' operation mode to monostatic...\n']);

%% Update Configuration

CFG.scanStartPs = uint32(scanStartPs);
CFG.scanStopPs = uint32(scanStopPs);
CFG.pulseIntegrationIndex = uint16(pulseIntegrationIndex);
CFG.transmitGain = uint8(transmitGain);
CFG.codeChannel = uint8(codeChannel);
CFG.antennaMode = uint8(antennaMode);
CFG.scanStepBins = uint16(scanStepBins);

% Send configuration
setConfig(s,1,CFG);

%% Read Configuration Confirmation

% Read packet
[msg, msgType, msgID] = readPacket(s);
if ~strcmp(msgType,'1101') % MRM_SET_CONFIG_CONFIRM
    error(fprintf('Invalid message type: %s, should have been 1101',msgType));
end

getConfig(s,1);
[msg, msgType, msgID] = readPacket(s);
CFG = parseMessage(msg);

fprintf(['Successfully updated node ' num2str(CFG.nodeId) ' configuration...\n']);

%% Formulate Control Structure

CTL.scanCount = uint16(scanCount);
CTL.reserved = uint16(0); % Aligns to word
CTL.scanIntervalTime = uint32(scanIntervalTime_ms*1000); % Microsecs between scan starts

%% Collect Data

% Initialize cell array that will contain information about each scan
rawCollect = cell(1,numScanLocs);

% Loop through scan locations
for i=1:numScanLocs
    
    fprintf('Please position radar at location %i (%2.2f, %2.2f, %2.2f) and press enter...\n',i,xLoc_m(i),yLoc_m(i),zLoc_m(i))
    pause

    % Request scan
    setControl(s,1,CTL);

    % Read confirm
    [msg, msgType, msgID] = readPacket(s);

    % Read first part of scan response
    [msg, msgType, msgID] = readPacket(s);

    % Determine number of following parts
    [scanInfo,msgType,msgID] = parseMessage(msg);

    % Save first part of raw scan
    scanRaw = double(scanInfo.scanData);

    % Save remaining parts of raw scan
    for j = 1:(scanInfo.numberMessages-1)
          [msg, msgType, msgID] = readPacket(s);
          [scanInfo,msgType,msgID] = parseMessage(msg);
          scanRaw = [scanRaw double(scanInfo.scanData)];
    end
    
    % Populate cell array
    rawCollect{i}.scan = scanRaw.';
    rawCollect{i}.xLoc_m = xLoc_m(i);
    rawCollect{i}.yLoc_m = yLoc_m(i);
    rawCollect{i}.zLoc_m = zLoc_m(i);
    rawCollect{i}.nodeID = CFG.nodeId;
    rawCollect{i}.scanStartPs = CFG.scanStartPs;
    rawCollect{i}.scanStopPs = CFG.scanStopPs;
    rawCollect{i}.scanResPs = scanResPs;
    rawCollect{i}.transmitGain = CFG.transmitGain;
    rawCollect{i}.antennaMode = CFG.antennaMode;
    rawCollect{i}.codeChannel = CFG.codeChannel;
    rawCollect{i}.pulseIntegrationIndex = CFG.pulseIntegrationIndex;
    rawCollect{i}.opMode = opMode;
    rawCollect{i}.distanceAxis_m = ([0:length(scanRaw)-1]*scanResPs/1e12)*C_mps/2;
    rawCollect{i}.scanIntervalTime_ms = scanIntervalTime_ms;
    
    fprintf('Completed scan at location %i...\n',i)    
    
    % Plot each scan as we go
    figure(1);
    hold all;
    plot(rawCollect{i}.distanceAxis_m,rawCollect{i}.scan);
    xlabel('Distance (m)');
    ylabel('Signal Strength');
    grid on;
    drawnow;

end

% Save to mat file
uisave('rawCollect','monoCollect.mat');

%% Close socket

fclose(s)
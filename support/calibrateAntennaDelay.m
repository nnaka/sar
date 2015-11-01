% MIT IAP 2013: Find a Needle in a Haystack
%
% Run this script to calibrate scanStartPs parameter for P410 antenna
% delay in monostatic mode.
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
fprintf('Monostatic Antenna Calibration Script\n');
fprintf('======================================================\n')

%% Initialize Constants

% User selectable
scanStartPs = 0;            % Minimum start scan point
scanStopPs = 100000;
pulseIntegrationIndex = 15; % The number of pulses per scan point (2^n)
transmitGain = 0;           % Tx power (0 for FCC legal)
scanIntervalTime_ms = 1;    % Time between start of each scan in millisecs

% Derived parameters
C_mps = 299792458;
codeChannel = 0;            % PN code
antennaMode = 2;            % Tx: A, Rx: B
scanStepBins = 32;          
scanResPs = scanStepBins*1.907; % 61 ps sampling resolution
scanCount = 1;              % number of scans (2^16-1 for continuous)

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
rawCollect = cell(1,1);

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
rawCollect{1}.scan = scanRaw.';
rawCollect{1}.nodeID = CFG.nodeId;
rawCollect{1}.scanStartPs = CFG.scanStartPs;
rawCollect{1}.scanStopPs = CFG.scanStopPs;
rawCollect{1}.scanResPs = scanResPs;
rawCollect{1}.transmitGain = CFG.transmitGain;
rawCollect{1}.antennaMode = CFG.antennaMode;
rawCollect{1}.codeChannel = CFG.codeChannel;
rawCollect{1}.pulseIntegrationIndex = CFG.pulseIntegrationIndex;
rawCollect{1}.opMode = opMode;
rawCollect{1}.distanceAxis_m = ([0:length(scanRaw)-1]*scanResPs/1e12)*C_mps/2;
rawCollect{1}.scanIntervalTime_ms = scanIntervalTime_ms;

% Plot each scan as we go
figure(1);
hold all;
plot((1:length(rawCollect{1}.scan))*rawCollect{1}.scanResPs,rawCollect{1}.scan);
xlabel('Time (ps)');
ylabel('Signal Strength');
grid on;
drawnow;


%% Close socket

fclose(s)
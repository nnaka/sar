

function [] = rangeTest()

%clear all;
%close all;
clc;
instrreset;
%warning off;

addpath('support');

fprintf('============================================\n')
fprintf('Monostatic Data Collection for Range Testing\n');
fprintf('  Tufts University | EE97 2014 | Blue Team\n');
fprintf('============================================\n')

%% Initialize Constants

% User selectable   
scanStartPs = 17400;            % Adjust this to match antenna delay
maxDistance_m = 13; %5;         % P410 will quantize to closest value above this number
pulseIntegrationIndex = 12; % The number of pulses per scan point (2^n)
transmitGain = 0;           % Tx power (0 for FCC legal)
scanIntervalTime_ms = 0;    % Time between start of each scan in millisecs
ovsFac = 4;                 % Oversampling factor applied when interpolating
rangeScaleFac = 3/2;        % Normalize by r^rangeScaleFac
ampFilter = 1.5e+7;           % Amplitude to filter below

% Derived parameters
C_mps = 299792458;
scanStopPs = scanStartPs + (2*maxDistance_m/C_mps)*1e12; % 1e12 ps in one sec
codeChannel = 0;            % PN code
antennaMode = 3;            % Tx: B, Rx: A
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

%% Initialize Collection Variables

% Initialize Signal Processing Constants
numSamp = length(rawCollect{1}.scan);               % Number of samples per scan
delta_r = scanResPs/1e12*C_mps/2;     % (m) Range resolution

% Pre-compute Variables
dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); % Ranges of each bin (m)
rangeScale = dataRange.^(rangeScaleFac); % Scale factor applied to the data as a fn of range
rngWin = hann_window(numSamp); % Window applied to reduce range sidelobes

% Initialize cell array that will contain information about each scan
rawCollect = cell(1,100);
input = ' ';
i = 1;

%% Collect Data

while input ~= ''
    
    fprintf('\n==============================\n')
    prompt1 = 'Press the space bar to collect a single scan\n or press enter to finish scanning.\n';
    input = input(prompt1,'c');

    %% Retrieve Raw Scan
    
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
    rawCollect{i}.distanceAxis_m = ([0:length(scanRaw)-1]*scanResPs/1e12)*C_mps/2;
    rawCollect{i}.xLoc_m = 0;
    rawCollect{i}.yLoc_m =0;
    rawCollect{i}.zLoc_m = 0;
    rawCollect{i}.nodeID = CFG.nodeId;
    rawCollect{i}.scanStartPs = CFG.scanStartPs;
    rawCollect{i}.scanStopPs = CFG.scanStopPs;
    rawCollect{i}.scanResPs = scanResPs;
    rawCollect{i}.transmitGain = CFG.transmitGain;
    rawCollect{i}.antennaMode = CFG.antennaMode;
    rawCollect{i}.codeChannel = CFG.codeChannel;
    rawCollect{i}.pulseIntegrationIndex = CFG.pulseIntegrationIndex;
    rawCollect{i}.opMode = opMode;
    rawCollect{i}.scanIntervalTime_ms = scanIntervalTime_ms;
    
    %% Process Raw Scan
    tmpRP = rawCollect{i}.scan;
    tmpRP = tmpRP.*rngWin;
    tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).'); % convert to IQ (hilbert transform then modulate to baseband)
    tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale; % interpolate up, and scale signal vs range  
    x = interp(rawCollect{i}.distanceAxis_m, ovsFac)';
    
    % Store distance of greatest reflection
    maxRP = abs(tmpRP);
    [~,I] = max(maxRP);
    rawCollect{i}.radarDistance = x(I)*3.2808;
    
    figure(1);
    hold all;
    plot(rawCollect{i}.distanceAxis_m,rawCollect{i}.scan);
    xlabel('Distance (m)');
    ylabel('Signal Strength');
    grid on;
    drawnow;
    movegui('northwest')

    figure(2);
    hold all;
    plot(x,abs(tmpRP));
    xlabel('Distance (m)');
    ylabel('Signal Strength');
    grid on;
    drawnow;
    movegui('northeast')
    
    % Store measured distance
    prompt2 = 'Enter measured distance in feet:\n';
    rawCollect{i}.measuredDistance = input(prompt2,'d');

end


% Save to mat file
uisave('rawCollect','monoCollect.mat');
% movegui('south')

% Close socket
fclose(s)

end
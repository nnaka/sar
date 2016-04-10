%% Initailize Parameters
clear;
filename = 'test.txt';
addpath('../data');

% User selectable   
scanStartPs = 17400;            % Adjust this to match antenna delay
% scanStopPs = 39297;
maxDistance_m = 20; %5;         % P410 will quantize to closest value above this number
pulseIntegrationIndex = 15; % The number of pulses per scan point (2^n)
transmitGain = 0;           % Tx power (0 for FCC legal)
scanIntervalTime_ms = 0;    % Time between start of each scan in millisecs
ovsFac = 4;                 % Oversampling factor applied when interpolating
rangeScaleFac = 3/2;        % Normalize by r^rangeScaleFac
ampFilter = 1.5e+7;           % Amplitude to filter below


% Derived parameters
C_mps = 299792458;
% maxDistance_m = C_mps * (scanStopPs - scanStartPs) / (2 * 1e12);
scanStopPs = scanStartPs + (2*maxDistance_m/C_mps)*1e12; % 1e12 ps in one sec
codeChannel = 0;            % PN code
antennaMode = 3;            % Tx: B, Rx: A
scanStepBins = 32;          
scanResPs = scanStepBins*1.907; % 61 ps sampling resolution
scanCount = 1;              % number of scans per spatial location (2^16-1 for continuous)    

%% Read Radar Data From File
[raw_scan gps_data] = readFile(filename);
scan_dim = size(raw_scan);               % [num_scans bins_per_scan]

%% Plot Raw Radar Data 
plotRawScan(raw_scan, scan_dim, scanResPs, C_mps);

%% Format Raw Radar Data
rawCollect = formatData(raw_scan, gps_data, other_metrics, scan_dim, ...
                        C_mps, scanResPs,scanIntervalTime_ms);

%% Process Raw Radar Data
display_on = true;                      % display image during processing?
height = 0.3810;                        % aperture height
monoSAR(rawCollect, maxDistance_m);
% pulse_set = processScan2D(rawCollect, ovsFac, C_mps, rangeScaleFac, ... 
%                           height, display_on);













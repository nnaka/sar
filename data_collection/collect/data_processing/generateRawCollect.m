function [rawCollect] = generateRawCollect(fileName)

% generateRawCollect
%
% reads a csv data file that contains raw radar returns and gps data
% and converts the data into a row vector of cells readable by SAR.m
%
% parameter:
%
% fileName - file name of csv file

% user selectable paramters
% values should match those declared in pulson.h
scanStepBins = 32;
scanResPs = scanStepBins * 1.907;   % 61 ps sampling resolution

%% Read Radar Data From File
[rawScan, gpsData] = readMultiScanFile(fileName);
scanDim = size(rawScan);

%% Process Raw Radar Data 
rawCollect = formatData(rawScan, gpsData, scanDim, scanResPs);

end
function [rawCollect] = formatData(rawScan, gpsData, scanDim, scanResPs)   

% formatData

% takes a vector of raw radar data and a vector of raw gps data 
% and outputs a single cell array 
%
% Parameters:
%
% rawScan   - 2D vector of radar returns. Each row consists of the radar
%             returns from a single pulse.
% gpsData   - 2D vector of GPS returns. Each row consists of the north, east,
%             down RTK data. Each row of GPS data corresponds to a row of 
%             radar data
% scanDim   - 1x2 matrix. Dimensions of rawScan. The first value is the
%             number of scans. The second is the number of data points per 
%             scan.
% ScanResPs - time interval between data points in each radar scan

C_mps = 299792458;
rawCollect = cell(scanDim(2), scanDim(1));

for i = 1:scanDim(1)
    rawCollect{i}.scan = rawScan(i,:).';
    rawCollect{i}.xLoc_m = gpsData(i,1);
    rawCollect{i}.yLoc_m = gpsData(i,2);
    rawCollect{i}.zLoc_m = gpsData(i,3);
    rawCollect{i}.scanResPs = scanResPs;
    rawCollect{i}.distanceAxis_m = ([0:length(rawScan(i,:))-1]...
                                    *scanResPs/1e12)*C_mps/2;
end 


end 

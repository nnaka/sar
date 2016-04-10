function [rawCollect] = formatData(rawscan, gps_data, other_metrics, ... 
                                   scan_dim, scanResPs, C_mps, ... 
                                   scanIntervalTime_ms)

rawCollect = cell(1, scan_dim(1));

for i = 1:scan_dim(1)
    rawCollect{i}.scan = rawscan(i,:).';
    rawCollect{i}.time = 0;
    rawCollect{i}.xLoc_m = gps_data(i,1);
    rawCollect{i}.yLoc_m = gps_data(i,2);
    rawCollect{i}.zLoc_m = gps_data(i,3);
    rawCollect{i}.nodeID = other_metrics(i,1);
    rawCollect{i}.scanStartPs = other_metrics(i,1);
    rawCollect{i}.scanStopPs = other_metrics(i,2);
    rawCollect{i}.scanResPs = other_metrics(i,3);
    rawCollect{i}.transmitGain = other_metrics(i,4);
    rawCollect{i}.antennaMode = other_metrics(i,5);
    rawCollect{i}.codeChannel = other_metrics(i,7);
    rawCollect{i}.pulseIntegrationIndex = other_metrics(i,6);
    rawCollect{i}.opMode = 1; %opMode;
    rawCollect{i}.scanIntervalTime_ms = scanIntervalTime_ms;
    rawCollect{i}.distanceAxis_m = ([0:length(rawscan(i,:))-1]...
                                    *scanResPs/1e12)*C_mps/2;
end 


end 
function [rawCollect] = formatData(rawscan, gps_data, scan_dim, ... 
                                   scanResPs, C_mps, scanIntervalTime_ms)

rawCollect = cell(scan_dim(3), scan_dim(1));
for i = 1:scan_dim(3)
        for j = 1:scan_dim(1)
        rawCollect{i,j}.scan = rawscan(j,:).';
        rawCollect{i,j}.time = 0;
        rawCollect{i,j}.xLoc_m = gps_data(j,1);
        rawCollect{i,j}.yLoc_m = gps_data(j,2);
        rawCollect{i,j}.zLoc_m = gps_data(j,3);
        rawCollect{i,j}.scanResPs = scanResPs;
        rawCollect{i,j}.scanIntervalTime_ms = scanIntervalTime_ms;
        rawCollect{i,j}.distanceAxis_m = ([0:length(rawscan(i,:))-1]...
                                        *scanResPs/1e12)*C_mps/2;
        end 
end


end 

%     rawCollect{i}.nodeID = other_metrics(i,1);
%     rawCollect{i}.scanStartPs = other_metrics(i,1);
%     rawCollect{i}.scanStopPs = other_metrics(i,2);
%     rawCollect{i}.transmitGain = other_metrics(i,4);
%     rawCollect{i}.antennaMode = other_metrics(i,5);
%     rawCollect{i}.codeChannel = other_metrics(i,7);
%     rawCollect{i}.pulseIntegrationIndex = other_metrics(i,6);
%     rawCollect{i}.opMode = 1; %opMode;
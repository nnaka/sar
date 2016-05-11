function [rawCollect] = formatData(rawscan, gps_data, scan_dim, ... 
                                   scanResPs)
                               
C_mps = 299792458;
rawCollect = cell(scan_dim(2), scan_dim(1));

for j = 1:scan_dim(1)
    rawCollect{j}.scan = rawscan(j,:).';
    rawCollect{j}.xLoc_m = gps_data(j,1);
    rawCollect{j}.yLoc_m = gps_data(j,2);
    rawCollect{j}.zLoc_m = gps_data(j,3);
    rawCollect{j}.scanResPs = scanResPs;
    rawCollect{j}.distanceAxis_m = ([0:length(rawscan(j,:))-1]...
                                    *scanResPs/1e12)*C_mps/2;
end 


end 

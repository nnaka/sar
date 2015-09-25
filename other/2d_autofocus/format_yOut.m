function [rawCollect] = format_yOut()

load('PulsOnTestData.mat');
numScans = length(yOut(1,:));

rawCollect = cell(1,numScans);
for i=1:numScans 
    rawCollect{i}.scan = real(yOut(:,i));
    
    
    rawCollect{i}.xLoc_m = 0;
    rawCollect{i}.yLoc_m =0;
    rawCollect{i}.zLoc_m = 0;
    rawCollect{i}.nodeID =0;
    rawCollect{i}.scanStartPs = 0;
    rawCollect{i}.scanStopPs = 0;
    rawCollect{i}.scanResPs = 800;
    rawCollect{i}.transmitGain = 0;
    rawCollect{i}.antennaMode = 0;
    rawCollect{i}.codeChannel = 0;
    rawCollect{i}.pulseIntegrationIndex =0;
    rawCollect{i}.opMode = 0;
    rawCollect{i}.scanIntervalTime_ms =0;
    
end

end
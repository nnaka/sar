function [rawCollect] = FormatPulseData(yOut)

% Format raw radar pulse returns from the FormPulses function such that
% they are compatible with SAR_3D 

fprintf('Formatting pulse data...\n');

numScans = length(yOut(1,:,1));
numRows = length(yOut(1,1,:));
rawCollect = cell(1,numScans);
for j = 1:numRows
    for i = 1:numScans
        
        rawCollect{j,i}.scan = real(yOut(:,i,j));
        rawCollect{j,i}.xLoc_m = 0;
        rawCollect{j,i}.yLoc_m = 0;
        rawCollect{j,i}.zLoc_m = 0;
        rawCollect{j,i}.nodeID = 0;
        rawCollect{j,i}.scanStartPs = 0;
        rawCollect{j,i}.scanStopPs = 0;
        rawCollect{j,i}.scanResPs = 600;
        rawCollect{j,i}.transmitGain = 0;
        rawCollect{j,i}.antennaMode = 0;
        rawCollect{j,i}.codeChannel = 0;
        rawCollect{j,i}.pulseIntegrationIndex = 0;
        rawCollect{j,i}.opMode = 0;
        rawCollect{j,i}.scanIntervalTime_ms = 0;

    end
end
end
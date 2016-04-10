function [pulse_set] = processScan2D(rawCollect, ovsFac, C_mps, ... 
                                     rangeScaleFac, height, display_on)

addpath('support');

%% Initialize constants TODO merge constant initialization to main file 
sceneSizeX = 20;       % (m) Scene extent in x-dimension (cross Range)
sceneSizeY = 20;       % (m) Scene extend in y-dimension
numPixelsX = 641;
numPixelsY = 481;
imgX = linspace(-sceneSizeX/2,sceneSizeX/2,numPixelsX).';
imgY = linspace(0,sceneSizeY,numPixelsY).';
imgZ = height;                          % (m) Altitude of 2-D backprojection image 

fc = 4.3e9;                             % (Hz) radar center frequency
numSamp = length(rawCollect{1}.scan);   % Number of samples per scan
numScans = length(rawCollect);          % Number of scans
angleLimit_deg = 90;                    % Angular limit of SAR image to form (each side)

%% Initialize Output Display
myImg = zeros(numel(imgY),numel(imgX));
if display_on
    hFig = figure;
    hAx = axes;
    hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
    currentPulse = zeros(size(myImg));
    colormap(hAx,jet(256));
    colorbar;
    xlabel('X (meters)');
    ylabel('Y (meters)');
    set(hAx,'YDir','normal');
end

%% Pre-compute variables

delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution

% Ranges of each bin (m)
dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); 

% Scale factor applied to the data as a fn of range
rangeScale = dataRange.^(rangeScaleFac); 

% Window applied to reduce range sidelobes
rngWin = hann_window(numSamp); 

if angleLimit_deg < 90
    for xIdx = 1:numel(imgX)
        % Determine which pixels are outside the angle limit of the image
        clipPixels = (abs(imgY) < abs(imgX(xIdx)*tan(pi/2 - angleLimit_deg*(pi/180))));
        % Set these out-of-bounds pixels to "unknown"
        myImg(clipPixels,xIdx) = nan;
    end
end

%% Process Scans
fprintf('Processing scans..\n');

pulseSet = zeros(numel(imgY),numel(imgX), numScans);

for scanIdx = 1:numScans
    
    currentPulse = zeros(size(myImg));
    
    tmpRP = rawCollect{scanIdx}.scan;
    tmpRP = tmpRP.*rngWin;
    
    % convert to IQ (hilbert transform then modulate to baseband)
    tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).');
    
    % interpolate up, and scale signal vs range
    tmpRP = fft_interp(tmpRP, ovsFac).*rangeScale;
    
    % compute the first difference in range (used for linear interpolation)
    diffRP = diff([tmpRP; 0],1);
    
    % incorporate this position into the image via backprojection
    for xIdx = 1:numel(imgX)
        
        % compute the range to each image pixel & the matched filter terms
        rangeVec = sqrt((imgX(xIdx) - rawCollect{scanIdx}.xLoc_m)^2 ...
                   + (imgZ - rawCollect{scanIdx}.zLoc_m)^2           ...
                   + (imgY - rawCollect{scanIdx}.yLoc_m).^2);
        matchVec = exp(1j * (2*pi*fc) * (2*rangeVec/C_mps)); % assumes narrow band
        
        % compute integer and fractioal range indices to each image pixel
        % (for linear interpolation)
        % find index in scan associated with distance
        rangeInds = rangeVec * (ovsFac / delta_r) + 1;
        rangeLo = floor(rangeInds);
        rangeFrac = rangeInds - rangeLo;
        
        % perform linear interpolation and apply the matched filter terms
        % (this is the backprojection integral)
        trimLen = sum(rangeLo>length(tmpRP));
        z = rangeLo(1:end-trimLen);
        
        currentPulse(:,xIdx) = [(tmpRP(z) + diffRP(z)      ... 
                               .* rangeFrac(1:end-trimLen))...
                               .* matchVec(1:end-trimLen); ...
                               zeros(trimLen,1)];
    
    end 
    
    rangeData(:,scanIdx) = tmpRP;
    pulseSet(:,:,scanIdx) = currentPulse;
    myImg = myImg + currentPulse;
    
    % to save current pulse and pulse updates:
    % save(strcat('pulse',strcat(int2str(scanIdx),'.mat')));
    
    % update the display
    if display_on
        img_dB = 20*log10(abs(myImg));
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf('%6.2f%% complete', 100*scanIdx/numScans));
        if ~isinf(max(img_dB));
            caxis(hAx,max(img_dB(:)) + [-20 0]);
        end
        drawnow;
    end 
end
    
    
    
    
    
    
    
%     for i = 1:length(rawCollect);
% 
%         tmpRP = rawCollect{i}.scan;
%         tmpRP = tmpRP.*rngWin;
% 
% 
%         tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:length(tmpRP)-1)).');
% 
% 
%         tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale;
%         x = interp(rawCollect{i}.distanceAxis_m, ovsFac)';
% 
%         % plot each scan incrimentally
%         hold on;
%         plot(x, abs(tmpRP));
%         xlabel('Distance (m)');
%         ylabel('Signal Strength');
%         grid on;
% 
%         pause(0.1);
% 
%     end
% 
% 
% 
% end 
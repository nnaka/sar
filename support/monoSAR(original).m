% MIT IAP 2013: Find a Needle in a Haystack
%
% Run this script to create backprojection SAR image from monostatic scans.
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

function [] = monoSAR(radarReturns, IRReturns)

clc;

addpath('support');

fprintf('======================================================\n')
fprintf('MIT IAP 2013: Find a Needle in a Haystack\n')
fprintf('Monostatic Backprojection SAR Script\n');
fprintf('======================================================\n')

%% Initialize Constants

% Load raw monostatic collection data
load(radarReturns);

if ~exist('rawCollect')
   error('No supported data structures found.') 
end

% User selectable
ovsFac = 4;                  % Oversampling factor applied when interpolating
sceneSizeX = 12;%6.6667;     % (m) Scene extent in x-dimension (cross-range)
angleLimit_deg = 45;         % Angular limit of the SAR image to form (each side)
rngWindowEnable = true;      % Window applied to reduce range sidelobes
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac
imgZ = 0.3810;               % (m) Altitude of 2-D backprojection image
ampFilter = 2e+7;            % Amplitude to filter below

% Static parameters
C_mps = 299792458;                                  % (m/s) Speed of light
fLow = 3100e6;                                      % (Hz) Waveform operating band
fHigh = 5300e6;                                     % (Hz) Waveform operating band
fc = (fLow + fHigh)/2;                              % (Hz) Waveform center frequency
numSamp = length(rawCollect{1}.scan);               % Number of samples per scan
numScans = length(rawCollect);                      % Number of scans
delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution
imgX = linspace(-sceneSizeX/2,sceneSizeX/2,641).';  % (m) Image pixels in x-dim
sceneSizeY = sceneSizeX * 3/4;                      % Aspect ratio for 640 x 480 grid
imgY = linspace(0,sceneSizeY,481).';                % (m) Image pixels in y-dim


%% Initialize Output Display

hFig = figure;
hAx = axes;
myImg = zeros(numel(imgY),numel(imgX));
hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
colormap(hAx,jet(256));
colorbar;
xlabel('X (meters)');
ylabel('Y (meters)');
set(hAx,'YDir','normal');
movegui('southwest')

%% Import IR position data
positionData = IRDataImport(IRReturns);

%% Match timestamps with position data
rawCollect = syncPosition(rawCollect, positionData);

%% Pre-compute Variables

dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); % Ranges of each bin (m)
rangeScale = dataRange.^(rangeScaleFac); % Scale factor applied to the data as a fn of range

if rngWindowEnable
	rngWin = hann_window(numSamp); % Window applied to reduce range sidelobes
else
	rngWin = ones(numSamp,1);
end

if angleLimit_deg < 90
    for xIdx = 1:numel(imgX)
        % Determine which pixels are outside the angle limit of the image
        clipPixels = (abs(imgY) < abs(imgX(xIdx)*tan(pi/2 - angleLimit_deg*(pi/180))));
        % Set these out-of-bounds pixels to "unknown"
        myImg(clipPixels,xIdx) = nan;
    end
end

%% Process Scans

fprintf('Processing scans...\n');

for scanIdx = 1:numScans

    % % Remove signal in first 5m
    % for j=1:numSamp
    %     if rawCollect{scanIdx}.distanceAxis_m(j) < 5
    %         rawCollect{scanIdx}.scan(j) = 0;
    %     end
    %      if rawCollect{scanIdx}.distanceAxis_m(j) > 10
    %         rawCollect{scanIdx}.scan(j) = 0;
    %     end
    % end
    
    tmpRP = rawCollect{scanIdx}.scan;
    
    tmpRP = tmpRP.*rngWin;

    tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).'); % convert to IQ (hilbert transform then modulate to baseband)
    
    tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale; % interpolate up, and scale signal vs range  
    
    % % Remove signal below ampFilter
    % for j=1:(numSamp*ovsFac)
    %     if tmpRP(j) < (ampFilter)
    %         tmpRP(j) = 0;
    %     end
    % end

    % compute the first difference in range (used for linear interpolation)
    diffRP = diff([tmpRP; 0],1);
    
%     figure;
    
    % incorporate this position into the image via backprojection
    for xIdx = 1:numel(imgX)
        % compute the range to each image pixel & the matched filter terms
        rangeVec = sqrt((imgX(xIdx) - rawCollect{scanIdx}.xLoc_m)^2 + (imgZ - rawCollect{scanIdx}.zLoc_m)^2 + (imgY - rawCollect{scanIdx}.yLoc_m).^2);
        matchVec = exp( 1j * (2*pi*fc) * (2*rangeVec/C_mps) ); % assumes narrowband
     
        
        % compute integer and fractional range indices to each image pixel (for linear interpolation)
        rangeInds = rangeVec * (ovsFac / delta_r) + 1;
        rangeLo = floor(rangeInds);
        rangeFrac = rangeInds - rangeLo;
        
%         plot(rangeLo);
%         shg;
        
        
        % perform linear interpolation and apply the matched filter terms
        %    (this is just the backprojection integral)
        trimLen = sum(rangeLo>length(tmpRP));

        z = rangeLo(1:end-trimLen);

        
        aa = tmpRP(z);
        ab = diffRP(z).*rangeFrac(1:end-trimLen) .* matchVec(1:end-trimLen);
        ac = zeros(trimLen,1);


        myImg(:,xIdx) = myImg(:,xIdx) + [(tmpRP(z) + diffRP(z).*rangeFrac(1:end-trimLen)) .* matchVec(1:end-trimLen); zeros(trimLen,1)];
    end
    
    % update the user display 
    img_dB = 20*log10(abs(myImg));
    set(hImg,'CData',img_dB);
    set(hFig,'Name',sprintf('%6.2f%% complete',100*scanIdx/numScans));
    if ~isinf(max(img_dB(:)))
        caxis(hAx,max(img_dB(:)) + [-20 0]);
    end
    drawnow;
    
end

% figure
% surf(imgX,imgY,20*log10(abs(myImg)))
% shading interp;
% xlabel('X (meters)','FontSize',12);
% ylabel('Y (meters)','FontSize',12);
% zlabel('Backprojection (dB)','FontSize',12);
% movegui('southeast')

end
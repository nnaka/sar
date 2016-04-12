%% monoSAR
% Produces 2D SAR images from 1D apertures
function [image_set] = SAR_2D(rawCollect, sceneSize, display_image, ...
                               GPS_override, scan_incriment)


ovsFac = 4;                  % Oversampling factor applied when interpolating
angleLimit_deg = 90;         % Angular limit of the SAR image to form (each side)
rngWindowEnable = true;      % Window applied to reduce range sidelobes
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac

% Static parameters
C_mps = 299792458;                                  % (m/s) Speed of light
fc = 4.3e9;                                         % (Hz) Waveform center frequency
numSamp = length(rawCollect{1}.scan);               % Number of samples per scan
numScans = length(rawCollect);                      % Number of scans
delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution

% Image parameters 
sceneSizeX = sceneSize(1);                          % (m) Scene extent in x-dimension (cross-range)
sceneSizeY = sceneSize(2);
%sceneSizeY = sceneSizeX * 3/4;                     % Aspect ratio for 640 x 480 grid

imgX = linspace(-sceneSizeX/2,sceneSizeX/2,641).';  % (m) Image pixels in x-dim
imgY = linspace(0,sceneSizeY,481).'; 
imgZ = sceneSize(3);                                % (m) Altitude of 2-D backprojection image
%% Initialize Output Display
myImg = zeros(numel(imgY),numel(imgX));
if display_image
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
%set(hFig, 'Position', [750 750 1500  1000])

%% Override GPS position data if requested 
if GPS_override
    aperture_len = scan_incriment * numScans;
%     xLoc = linspace(-aperture_len/2,aperture_len/2,numScans);
    xLoc = linspace(-(0.015*length(rawCollect))/2,(0.015*length(rawCollect))/2,length(rawCollect));

    for i=1:length(rawCollect)
%         rawCollect{i}.xLoc_m = (-scan_incriment*(i-1));
        rawCollect{i}.xLoc_m = xLoc(end-i+1);
        rawCollect{i}.yLoc_m = 0;
        rawCollect{i}.zLoc_m = 0;      % maybe???
    end
end

%% Pre-compute Variables
% Convert scan 'bins' from picoseconds to distance in meters (d=rt)
% Ranges of each bin (m)
dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); 

% Scale factor applied to the data as a fn of range
rangeScale = dataRange.^(rangeScaleFac); 

% Window applied to reduce range sidelobes
if rngWindowEnable
	rngWin = hann_window(numSamp); 
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

pulseSet = zeros(numel(imgY),numel(imgX),numScans);
for scanIdx = 1:numScans
    currentPulse = zeros(size(myImg));
    tmpRP = rawCollect{scanIdx}.scan;
    
    tmpRP = tmpRP.*rngWin;

    tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).'); % convert to IQ (hilbert transform then modulate to baseband)
    
    tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale; % interpolate up, and scale signal vs range

    % compute the first difference in range (used for linear interpolation)
    diffRP = diff([tmpRP; 0],1);
    
 
    % incorporate this position into the image via backprojection
    for xIdx = 1:numel(imgX)
        % compute the range to each image pixel & the matched filter terms
        rangeVec = sqrt((imgX(xIdx) - rawCollect{scanIdx}.xLoc_m)^2 + (imgZ - rawCollect{scanIdx}.zLoc_m)^2 + (imgY - rawCollect{scanIdx}.yLoc_m).^2);
        matchVec = exp( 1j * (2*pi*fc) * (2*rangeVec/C_mps) ); % assumes narrowband
     
        
        % compute integer and fractional range indices to each image pixel (for linear interpolation)
        rangeInds = rangeVec * (ovsFac / delta_r) + 1; % find index in scan associated with distance
        rangeLo = floor(rangeInds);
        rangeFrac = rangeInds - rangeLo;
        
        % perform linear interpolation and apply the matched filter terms
        %    (this is just the backprojection integral)
        trimLen = sum(rangeLo>length(tmpRP));

        z = rangeLo(1:end-trimLen);
  
        currentPulse(:,xIdx) = [(tmpRP(z) + diffRP(z).*rangeFrac(1:end-trimLen)) .* matchVec(1:end-trimLen); zeros(trimLen,1)];
        %myImg(:,xIdx) = myImg(:,xIdx) + [(tmpRP(z) + diffRP(z).*rangeFrac(1:end-trimLen)) .* matchVec(1:end-trimLen); zeros(trimLen,1)];
        
    end  
    rangeData(:,scanIdx) = tmpRP;
    pulseSet(:,:,scanIdx) = currentPulse;
    myImg = myImg + currentPulse;
    %save(strcat('pulse',strcat(int2str(scanIdx),'.mat')));
    % update the user display 
    if display_image
        img_dB = 20*log10(abs(myImg));
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf('%6.2f%% complete',100*scanIdx/numScans));
        if ~isinf(max(img_dB(:)))
            caxis(hAx,max(img_dB(:)) + [-20 0]);
        end
        drawnow;
    end
    
end
% if displayImage
%     close(hFig);
image_set = myImg;
end



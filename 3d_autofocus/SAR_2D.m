% -----------------------------------------------------------------------------
% Uses backprojection to form a 2D image with the given radar data 
% 
% @param rawCollect [Array] the formatted radar pulse data
% @param imgSize [Array] the voxel dimensions of the image [X Y]
% @param sceneSize [Array] the area that the image covers in meters [X Y Z]
% @return image [Array] 2D complex valued SAR image
% @return pulseHistory [Array] 3D array with phase history for autofocus
% -----------------------------------------------------------------------------
function [image, pulseHistory] = SAR_2D( rawCollect, imgSize, sceneSize )


ovsFac = 4;                  % Oversampling factor applied when interpolating
angleLimit_deg = 90;         % Angular limit of the SAR image to form (each side)
rngWindowEnable = true;      % Window applied to reduce range sidelobes
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac

% Static parameters
C_mps = 299792458;                                  % (m/s) Speed of light
fc = 4.3e9;                                         % (Hz) Waveform center frequency
numSamp = length(rawCollect{1}.scan);               % Number of samples per scan
numScans = numel(rawCollect);                       % Number of scans
delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution

voxelsX = imgSize(1);                              % Set image size
voxelsY = imgSize(2);

% Image parameters 
sceneSizeX = sceneSize(1);                          % (m) Scene extent in x-dimension (cross-range)
sceneSizeY = sceneSize(2);

imgX = linspace(-sceneSizeX/2,sceneSizeX/2,voxelsX).';
imgY = linspace(0,sceneSizeY,voxelsY).';
imgZ = sceneSize(3);                                % (m) Altitude of 2-D backprojection image

%% Initialize Output Display
pulseHistory = zeros(numel(imgY),numel(imgX),numScans);
image = zeros(numel(imgY),numel(imgX));

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
        image(clipPixels,xIdx) = nan;
    end
end

%% Process Scans

fprintf('Processing scans...\n');

for scanIdx = 1:numScans
    currentPulse = zeros(size(image));
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
    end  

    fprintf('scan %d of %d\n', scanIdx, numScans);

    rangeData(:,scanIdx) = tmpRP;
    pulseHistory(:,:,scanIdx) = currentPulse;
    image = image + currentPulse;
end

end



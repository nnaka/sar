% -----------------------------------------------------------------------------
% Uses backprojection to form a 3D image with the given radar data 
% 
% @param rawCollect [Array] the formatted radar pulse data
% @param imgSize [Array] the voxel dimensions of the image [X Y Z]
% @param sceneSize [Array] the area that the image covers in meters [X Y Z]
% @return image [Array] 3D complex valued SAR image
% @return pulseHistory [Array] 4D array with phase history for autofocus
% -----------------------------------------------------------------------------
function [image, pulseHistory] = SAR_3D( rawCollect, imgSize, sceneSize )

%% Initialize parameters 
addpath('support');

% User selectable
ovsFac = 4;                  % Oversampling factor applied when interpolating
angleLimit_deg = 90;         % Angular limit of the SAR image to form (each side)
rngWindowEnable = true;      % Window applied to reduce range sidelobes
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac

% Static parameters
C_mps                     = 299792458;                              % (m/s) Speed of light
fLow                      = 3100e6;                                 % (Hz) Waveform operating band
fHigh                     = 5300e6;                                 % (Hz) Waveform operating band
fc                        = (fLow + fHigh)/2;                       % (Hz) Waveform center frequency
numSamp                   = length(rawCollect{1,1}.scan);           % Number of samples per scan
collectSize               = size(rawCollect);
numScans                  = collectSize(2);
numRows                   = collectSize(1);
delta_r                   = rawCollect{1,1}.scanResPs/1e12*C_mps/2; % (m) Range resolution

voxelsX = imgSize(1);                              % Set image size
voxelsY = imgSize(2);
voxelsZ = imgSize(3);

sceneSizeX = sceneSize(1);                         % Set scene size 
sceneSizeY = sceneSize(2);
sceneSizeZ = sceneSize(3);

imgX = linspace(-sceneSizeX/2,sceneSizeX/2,voxelsX).';
imgY = linspace(0,sceneSizeY,voxelsY).';
imgZ = linspace(-sceneSizeZ/2,sceneSizeZ/2,voxelsZ).';


pulseHistory = zeros(voxelsY, voxelsX, voxelsZ, numRows*numScans);
image = zeros(voxelsY, voxelsX, voxelsZ);

%% Pre-compute Variables
% Convert scan 'bins' from picoseconds to distance in meters (d=rt)
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
        % This angle limit models the angle limit on our real-life antenna  
        clipPixels = (abs(imgY) < abs(imgX(xIdx)*tan(pi/2 - angleLimit_deg*(pi/180))));

        % Set these out-of-bounds pixels to "unknown"
        image(clipPixels,xIdx) = nan;
    end
end

%% Process Scans

fprintf('Processing scans...\n');

% Cross range windows to limit image sidelobes 
crsRngX = kaiser(numScans,3);
crsRngZ = kaiser(numRows,3);

k = zeros(1,1,numel(imgZ));

for rowIdx = 1:numRows
    for scanIdx = 1:numScans

        tmpRP = rawCollect{rowIdx,scanIdx}.scan;

        % apply range window 
        tmpRP = tmpRP.*rngWin;      

        % convert to IQ (hilbert transform then modulate to baseband)
        tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).');

        % interpolate up, and scale signal vs range
        tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale;                

        % compute the first difference in range (used for linear interpolation)
        diffRP = diff([tmpRP; 0],1);

        temp = rawCollect{rowIdx,scanIdx};              % temp variable because indexing is extremely slow 

        % incorporate this position into the image via backprojection
        for xIdx = 1:voxelsX
            % compute the range to each image pixel & the matched filter terms
            for yIdx = 1:voxelsY
                rangeVec = sqrt((imgX(xIdx) - temp.xLoc_m)^2 +...
                (imgZ - temp.zLoc_m).^2 +...
                (imgY(yIdx) - temp.yLoc_m)^2);

                matchVec = exp(1j * (2*pi*fc) * (2*rangeVec/C_mps)); % assumes narrowband

                % compute integer and fractional range indices to each image pixel (for linear interpolation)
                rangeInds = rangeVec * (ovsFac / delta_r) + 1; % find index in scan associated with distance
                rangeLo = floor(rangeInds);
                rangeFrac = rangeInds - rangeLo;

                % trim the data that falls outside of the range profile
                trimLen = sum(rangeLo>length(tmpRP));   
                z = rangeLo(1:end-trimLen);

                k(1,1,:) = crsRngX(scanIdx)*crsRngZ(rowIdx)*[((tmpRP(z) + diffRP(z)).*rangeFrac(1:end-trimLen) .* matchVec(1:end-trimLen)); zeros(trimLen,1)];                

                pulseHistory(yIdx, xIdx, :, scanIdx + numScans*(rowIdx-1)) = k;
                image(yIdx,xIdx,:) = image(yIdx,xIdx,:) + k;
            end
        end

        fprintf('scan %d of %d\n', scanIdx + (rowIdx-1)*numScans, numScans*numRows);
    end
end

end

function [image_set] = SAR_3D(rawCollect, img_size, scene_size, form_pulse_set)

% ----------------------------------------------------------------------- %
% SAR 3D 
%
% Uses backprojection to form a 3D image with the given radar data 
% 
% Inputs:
%   rawCollect      - the formatted radar pulse data
%   img_size        - the voxel dimensions of the image [X Y Z]
%   scene_size      - the area that the image covers in meters [X Y Z]
%   form_pulse_set  - boolean; specifies the creation of 4D pulse set 
%                              for the use with the autofocus routine
%
% ----------------------------------------------------------------------- %


%% Initialize parameters 
addpath('support');

% User selectable
ovsFac = 4;                  % Oversampling factor applied when interpolating
angleLimit_deg = 90;         % Angular limit of the SAR image to form (each side)
rngWindowEnable = true;      % Window applied to reduce range sidelobes
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac

% Static parameters
C_mps = 299792458;                                  % (m/s) Speed of light
fLow = 3100e6;                                      % (Hz) Waveform operating band
fHigh = 5300e6;                                     % (Hz) Waveform operating band
fc = (fLow + fHigh)/2;                              % (Hz) Waveform center frequency
numSamp = length(rawCollect{1,1}.scan);             % Number of samples per scan
collectSize = (size(rawCollect));
numScans = collectSize(2);                          
numRows = collectSize(1);
rawCollect{1,1}.scanResPs = 600;
delta_r = rawCollect{1,1}.scanResPs/1e12*C_mps/2;   % (m) Range resolution

voxelsX = img_size(1);                              % Set image size
voxelsY = img_size(2);
voxelsZ = img_size(3);

sceneSizeX = scene_size(1);                         % Set scene size 
sceneSizeY = scene_size(2);
sceneSizeZ = scene_size(3);

imgX = linspace(-sceneSizeX/2,sceneSizeX/2,voxelsX).';
imgY = linspace(0,sceneSizeY,voxelsY).';
imgZ = linspace(-sceneSizeZ/2,sceneSizeZ/2,voxelsZ).';


% Initialize the 4D pulse set if specified
% Otherwise, initialize the 3D data cube
% Only 4D data can be used with the autofocus routine 
if form_pulse_set
    pulseSet = zeros(voxelsY, voxelsX, voxelsZ, numRows*numScans);
else
    myCube = zeros(voxelsY, voxelsX, voxelsZ);
end

%% Set the position of each pulse within the aperture 

delta = C_mps / (4 * fc);                           % pulses are taken at 
for j = 1:numRows                                   % distances lambda / 4 
    for i=1:numScans                                % apart
        rawCollect{j,i}.xLoc_m = (delta*(i-1));
        rawCollect{j,i}.yLoc_m = 0;
        rawCollect{j,i}.zLoc_m = (delta*(j-1));
    end
end

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
        myImg(clipPixels,xIdx) = nan;
    end
end

%% Process Scans

% Cross range windows to limit image sidelobes 
crsRngX = kaiser(numScans,3);
crsRngZ = kaiser(numRows,3);

k = zeros(1,1,numel(imgZ));

fprintf('Processing scans...\n');
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
                
                % Form the 4D pulse set data if specified 
                % Otherwise, form a 3D image cube
                % Only 4D data can be used with the autofocus routine 
                if form_pulse_set
                    pulseSet(yIdx, xIdx, :, scanIdx + numScans*(rowIdx-1)) = k;
                else 
                    myCube(yIdx,xIdx,:) = myCube(yIdx,xIdx,:) + k;
                end
            end
        end

        fprintf('scan %d of %d\n', scanIdx + (rowIdx-1)*numScans, numScans*numRows);
    end
end

% output the pulse set if creation of the 4D dataset has been specified
if form_pulse_set
    image_set = pulseSet;
else 
    image_set = myCube;
end 

end

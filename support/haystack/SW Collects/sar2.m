function [] = sar2(wavFile)
% function [] = cantenna_sar_v3(wavFile)
% 
% Produces a synthetic aperture image of the cantenna
% recording. This processing assumes the data was 
% collected by switching the radar on/off at several
% positions. See inside the function for more
% parameters.
%
%    wavFile = the filename of the .WAV file to process
%
% Lincoln Laboratory MIT IAP Radar Course
% (c) 2012 Massachusetts Institute of Technology

% the following line avoids a stylistic warning from Matlab
%#ok<*UNRCH>

% ---- setup constants and parameters ----
c = 3e8; % (m/s) speed of light
Tp = 10e-3; % (s) minimum pulse length
Trp = .1; % (s) minimum range profile duration

%delta_x = 2*(1/12)*0.3048; % (m) 2 inch antenna spacing, collect left-to-right
%delta_x = -2 * unitsratio('m','in'); % (m) 2-inch antenna spacing, collect right-to-left
% delta_x = 2 * unitsratio('m','in'); % (m) 2-inch antenna spacing, collect left-to-right
delta_x = .05; % (m) 2-inch antenna spacing, collect left-to-right

numPad = 64; % number of samples to pad for bandlimited interpolation & shifting
ovsFac = 16; % oversampling factor applied when interpolating the trigger & data signals
sceneSizeX = 140; % (m) scene extent in x-dimension (cross-range)
angleLimit_deg = 45; % angular limit of the SAR image to form (each side)
useSpatialTaper = false; % use a "spatial taper" to reduce angle sidelobes?

% LFM start-stop frequency (Hz) 
% use for data 5/9 -- wider band, using peaks, measured 5/9/12
fStart = 2295e6; 
fStop = 2542e6;

% ----- end constants and parameters -----

% use a default filename if none is given
if ~exist('wavFile','var')
    wavFile = '../data/towardswarehouse.wav';
end

% read the raw wave data
fprintf('Using %g MHz bandwidth\n', (fStop-fStart)*1e-6);
fprintf('Loading WAV file...\n');
[Y,Fs] = wavread(wavFile,'native');

% derived parameters
Np = round(Tp * Fs); % # of samples per pulse
Nrp = round(Trp * Fs); % minimum # of samples per range profile
BW = fStop - fStart; % (Hz) transmit bandwidth
delta_r = c/(2*BW); % (m) range resolution
fc = (fStart + fStop)/2; % (Hz) the center frequency
imgX = linspace(-sceneSizeX/2,sceneSizeX/2,641).'; % (m) image pixels in x-dim
sceneSizeY = sceneSizeX * 3/4; % so the result looks good on a 640 x 480 grid
imgY = linspace(0,sceneSizeY,481).'; % (m) image pixels in y-dim

% change sign of the input because it appears inverted in practice
trig = -Y(:,1); % the trigger signal is in the first channel
s = -Y(:,2); % the mixer output is in the second channel
clear Y;

% parse the trigger signal (look for silences & for threshold crossings)
fprintf('Parsing the recording...\n');
%
breakTrig = (abs(trig) > mean(abs(trig)));
pulseTrig = (trig > 0);
%
breakSumL = sum(breakTrig(1:Nrp));
breakSumR = sum(breakTrig((Nrp+1)+(1:Nrp)));
pulseSum = sum(pulseTrig((Nrp-Np+1):Nrp));
%
pulseStarts = [];
breakNumber = [];
plsPerBreak = 0;
currentBreak = 0;
inBreak = false;
%
for ii = Nrp+1:numel(trig)-Nrp-1
    if (breakTrig(ii) && breakSumL==0)
        currentBreak = currentBreak + 1;
        plsPerBreak = [plsPerBreak; 0]; %#ok<*AGROW>
        inBreak = false;
    end
    inBreak = inBreak | (breakSumR==0);
        
    if ~inBreak && (pulseTrig(ii) && pulseSum==0)
        pulseStarts = [pulseStarts; ii];
        breakNumber = [breakNumber; currentBreak];
        plsPerBreak(end) = plsPerBreak(end) + 1;
    end        
   
    % update the running sums
    breakSumL = breakSumL + breakTrig(ii) - breakTrig(ii - Nrp);
    breakSumR = breakSumR - breakTrig(ii) + breakTrig(ii + Nrp + 1);
    pulseSum = pulseSum + pulseTrig(ii) - pulseTrig(ii - Np);
end
%
clear breakTrig pulseTrig;
%
numBreaks = currentBreak;
numPulses = numel(pulseStarts);
fprintf('Found %d breaks and %d pulses\n',numBreaks,numPulses);

% refine using measured parameters: the pulse width
Np = round(max(diff(pulseStarts))/2);
for bIdx = 2:numBreaks
    myPulses = find(breakNumber == bIdx);
    Np = min(min(round(diff(pulseStarts(myPulses(2:end-1)))/2)),Np);
end
Tp = Np / Fs;
fprintf('Measured pulse width of %g ms \n', Tp*1e3);
Npi=Np;
Npm=Np;
Npx=Np;
for bIdx = 2:numBreaks
    myPulses = find(breakNumber == bIdx);
    Npi = min(min(round(diff(pulseStarts(myPulses(2:end-1)))/2)),Npi);
    Npm = median(median(round(diff(pulseStarts(myPulses(2:end-1)))/2)),Npm);
    Npx = max(max(round(diff(pulseStarts(myPulses(2:end-1)))/2)),Npx);
end
fprintf('Measured minmum pulse width of %g ms, max %g, median %g \n', (Npi/Fs)*1e3, (Npx/Fs)*1e3, (Npm/Fs)*1e3);
Np = Npm; % use median for more accurate range axes
% Tp = Np / Fs; % still needed??

% we have to use the same # of pulses per break, so we are limited to the min
%   the data  before the first break may be incomplete, so ignore it
minPulsesPerBreak = min(plsPerBreak(2:end));
fprintf('Minimum of %d pulses between breaks\n',minPulsesPerBreak);

% compute aperture, range, & cross-range dimensions
Xa = ((1:(numBreaks-1)).' - (numBreaks-2)/2) * delta_x; % (m) cross range position of radar on aperture L
Ya = 0*Xa; % (m) range position of radar, assumes aperture is along a straight line
Za = 0*Xa; % (m) altitude position of radar, assumes aperture is at 0 m elevation relative to the scene

% make figure for output display
hFig = figure;
hAx = axes;
myImg = zeros(numel(imgY),numel(imgX));
hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
colormap(hAx,jet(256));
colorbar;
xlabel('X (meters)');
ylabel('Y (meters)');
set(hAx,'YDir','normal');

% pre-compute some windows and other vectors 
dataRange = (0:ovsFac*Np-1).' * (delta_r/ovsFac); % ranges of each bin (m)
rangeScale = dataRange.^(3/2); % scale factor applied to the data as a fn of range
%
rngWin = hann_window(Np); % the window applied to reduce range sidelobes
padWin = sin( (1:numPad).'/(numPad+1) * pi/2) .^2; % the window applied to the padded data
trgWin = hann_window(numPad*2+1); % the window applied to the trigger data
%
slowWin = ones(numBreaks-1,1); % the window applied to the slow-time (cross-range) data
if useSpatialTaper
    slowWin = hann_window(numBreaks-1); % user requested a hann window
end
%
carrierPhase = exp( -1j * (4*pi*fc/c * delta_r) * (0:Np-1).' ); % the (residual) carrier phase of each range bin
%
if angleLimit_deg < 90
    for xIdx = 1:numel(imgX)
        % determine which pixels are outside the angle limit of the image
        clipPixels = (abs(imgY) < abs(imgX(xIdx)*tan(pi/2 - angleLimit_deg*(pi/180))));
        % set these out-of-bounds pixels to "unknown"
        myImg(clipPixels,xIdx) = nan;
    end
end

% loop over break & process pulses
fprintf('Processing position data...\n');
for bIdx = 2:numBreaks
    posIdx = bIdx - 1; % the position index is the break index - 1
    myPulses = find(breakNumber == bIdx); % all pulses for this position
    
    % compute the zero-doppler mixer output for this position
    tmpRP = zeros(Np,1);
    for pIdx = 2:minPulsesPerBreak-1
        % bandlimited interpolate the trigger signal
        tmp = double(trig(pulseStarts(myPulses(pIdx)) + (-numPad:numPad))) .* trgWin;
        interpTmp = fft_interp(tmp,ovsFac);
        interpTmp = interpTmp( (numPad*ovsFac + 1) + (-ovsFac:ovsFac) );
        interpOffs = (-ovsFac:ovsFac)/ovsFac;
        myIdx = find(diff(sign(interpTmp))==2)+1;
        tmp2 = interpTmp( myIdx + (-1:0) );
        % linear interpolate to find the zero crossing
        fracOffset = -(interpOffs(myIdx) - tmp2(2)/(tmp2(2)-tmp2(1)) / ovsFac);
        
        % time-align the data to the trigger event (the zero crossing)
        cInds = pulseStarts(myPulses(pIdx)) + (-numPad:(Np+numPad-1));
        tmp = double(s(cInds));
        tmp(1:numPad) = tmp(1:numPad) .* padWin;
        tmp(end:-1:(end-numPad+1)) = tmp(end:-1:(end-numPad+1)) .* padWin;
        % time delay applied in the frequency domain below
        tmp = fft(tmp);
        tmp = tmp .* exp( -1j*(0:(Np+2*numPad-1)).'/(Np+2*numPad)*2*pi*fracOffset );
        tmp = ifft(tmp,'symmetric');
        
        % incorporate this pulse in the average (is computing the zero-doppler mixer output)
        tmpRP = tmpRP + tmp(numPad + (1:Np));
    end
    
    % compute the range profile from the mixer output
    tmpRP = ifft(tmpRP .* (rngWin*slowWin(posIdx))); % apply fast & slow-time windows, then ifft
    tmpRP = fft_interp(tmpRP .* carrierPhase, ovsFac) .* rangeScale; % baseband (remove carrier phase), interpolate up, and scale signal vs range
    
    % compute the first difference in range (used for linear interpolation)
    diffRP = diff([tmpRP; 0],1);
    
    % incorporate this position into the image via backprojection
    for xIdx = 1:numel(imgX)
        % compute the range to each image pixel & the matched filter terms
        rangeVec = sqrt(((imgX(xIdx) - Xa(posIdx))^2 + Za(posIdx)^2) + (imgY - Ya(posIdx)).^2);
        matchVec = exp( 1j * (4*pi*fc/c) * rangeVec );
        
        % compute integer and fractional range indices to each image pixel (for linear interpolation)
        rangeInds = rangeVec * (ovsFac / delta_r) + 1;
        rangeLo = floor(rangeInds);
        rangeFrac = rangeInds - rangeLo;
        
        % perform linear interpolation and apply the matched filter terms
        %    (this is just the backprojection integral)
        myImg(:,xIdx) = myImg(:,xIdx) + (tmpRP(rangeLo) + diffRP(rangeLo).*rangeFrac) .* matchVec;
    end
    
    % update the user display 
    img_dB = 20*log10(abs(myImg));
    set(hImg,'CData',img_dB);
    set(hFig,'Name',sprintf('%6.2f%% complete',100*posIdx/(numBreaks-1)));
    caxis(hAx,max(img_dB(:)) + [-40 0]);
    drawnow;
end
title(wavFile,'interpreter','none');
boldify2;

% ---- standard DSP helper functions below ----

function [y] = fft_interp(x,M)
% perform approximate bandlimited interpolation of x by a factor of M
L = 4;
winInds = (-L*M : L*M).'/M * pi;

% get the ideal antialiasing filter's impulse response of length 2*M + 1 
winInds(L*M + 1) = 1;
myWin = sin(winInds) ./ winInds;
myWin(L*M + 1) = 1;

% use the window method; apply a hann window
myWin = myWin .* hann_window(2*L*M + 1);

% insert zeros in data and apply antialias filter via FFT
nFFT = numel(x) * M;
if isreal(x)
    y = ifft( fft(myWin,nFFT) .* repmat(fft(x),[M 1]), 'symmetric');
else
    y = ifft( fft(myWin,nFFT) .* repmat(fft(x),[M 1]) );
end
y = y([L*M+1:end 1:L*M]);

function [w] = hann_window(N)
% create a hann (cosine squared) window
w = .5 + .5*cos(2*pi*((1:N).'/(N+1) - .5));

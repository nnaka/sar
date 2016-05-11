% -----------------------------------------------------------------------------
% Forms an image using the parameters set below and saves it to the specified
% filename
%
% 1 - Generates a set of radar point reflectors 
% 2 - Forms pulsed radar data from these points
%
% Current Limitations:
%   If you are trying to create a pulseSet to run with autofocus with an
%   aperture size greater than 40x40 pulses and image size greater than
%   50x50x50 voxels , MATLAB will consume every bit of RAM and implode. 
%   Intensive memory use is a HUGE issue here. Unless your computer is a 
%   computational beast with limitless RAM, it will run out of storage. 
%
%   If you are thinking, wait, a 40x40 pulse aperture is many orders 
%   magnitude smaller than the ideal aperture size, you are correct. This 
%   memory problem will grow exponentially as we increase aperture sizes to
%   our operational goal of 30x5m with ~110,000 pulses. 
%
% Also of note:
%   With current numbers for GPS error (std X, Y = 0.0051m; Z = 0.0071m)
%   GPS error is unnoticable in images created from apertures upwards of
%   100x100 pulses, and barely noticable in apertures around 40x40 pulses. 
%
% @return rawCollect [Struct] the formatted radar pulse data
% -----------------------------------------------------------------------------
function [rawCollect] = formImage()

%% Generate target points 

% These vectors define the locations of the set of point source scatterers that
% we model in the scenario.  so long as each point has an [X Y Z] coordinate,
% there can be any number of points in any arrangement that you choose...have
% fun!

x0=kron((-20:20:20)',ones(3,1));     
y0=kron(ones(3,1),(-20:20:20)');
z0=[0;30;-30;30;0;-30;30;-30;0];

% Some other examples

% x0 = 10;
% y0 = 10;
% z0 = 10;

% x0 = linspace(-5,5,100)';
% y0 = zeros(numel(x0),1);
% z0 = y0;

% x0 = [-10,0,10];
% y0 = [-10,0,10];
% z0 = [-10,0,10];

%x0 = [-30,-30,-30,-30,-30,-20,-10,-10,-10,-10,-10,10,20,30,20,20,20,10,20,30];
%y0 = [10,5,0,-5,-10,0,10,5,0,-5,-10,10,10,10,5,0,-5,-10,-10,-10];
%z0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'

nX       = 20;      % # of points per row in synthetic aperture
nY       = 20;      % # of rows in synthetic aperture
noiseVec = [0,0,0]; % nonzero values will inject noise

%% Generate SAR data with the given points 
pulseData = formPulses(x0, y0, z0, nX, nY, noiseVec);

% For testing purposes, we do not generate new data each run, and instead
% load formatted data from a file.

%% Format the raw radar data 
rawCollect = formatPulseData(pulseData);

end

% -----------------------------------------------------------------------------
% Form Pulses
% 
% With given target locations and pulses in the aperture, this function
% will generate artificial radar pulses of the given scene. Designed to
% simulate the PulsOn p410 radar unit as accurately as possible. This
% additionally saves the data to RawImageData.mat.
%
% Inputs:
%   x0, y0, z0     - [ X Y Z ] coordinates of each radar target
%   nX             - number of pulses per row of the aperture
%   nY             - number of rows in the aperture
%   noiseFactorVec - noise multiplier [x,y,z] -- default noise: 10 mm
%
% -----------------------------------------------------------------------------
function [scans] = formPulses(x0, y0, z0, nX, nY, noiseFactorVec)

% Initialize Constants 
maxRange = 1e3;                 % max radar range in meters
c = 299792458;                  % speed of light
fLow = 3100e6;                  % (Hz) Waveform operating band
fHigh = 5300e6;                 % (Hz) Waveform operating band
fc = (fLow + fHigh)/2;          % (Hz) Waveform center frequency
fs = 1.25e9;                    % (Hz) radar sample frequency
droneVel = 1;                   % drone speed in m/s
lambda = c/fc;                  % pulse wavelength
prf = droneVel * 2 / lambda;    % pulse repitition frequency
scaling = (maxRange*2)/c;
tauP = ones(nX) * scaling;      % listening time

% Initialize Aperture 
step_size = lambda / 4;         % radar pulses taken every lambda / 4 meters
yCent = -80;
xCent = 0; xStep = step_size;
zCent = 0; zStep = step_size;

for n = 1:nY
    x=xCent+xStep*(-(nX-1)/2:1:(nX-1)/2);  % radar positions
    y=yCent*ones(size(x));
    z=zCent+zStep*(n-(nY-1)/2)*ones(size(x));
     
    % noise is a normal curve in each dimension and is based on the expected
    % error from the RTK GPS unit

    % GPS AVERAGES

    meanX  = 0;             
    meanY  = -0.009;
    meanZ  = -0.0083;

    sigmaX = 0.0051;        
    sigmaY = 0.0051;         
    sigmaZ = 0.0071;

    % Further testing needs to be completed to determine the meanX and sigmaX...
    % X position is a function of time; GPS data needs to be correlated with
    % video frames sigmaX assumed to be equal to sigmaY because SwiftNav claims
    % error distributions are equal in the xy plane

    % Add random noise into the aperture positions;
    % Noise was experimentally found to be a normal distrubution curve

    randVec = normrnd(meanX,sigmaX,1,length(x));
    x = x + randVec * noiseFactorVec(1);

    randVec = normrnd(meanY,sigmaY,1,length(y));
    y = y + randVec * noiseFactorVec(2);

    randVec = normrnd(meanZ,sigmaZ,1,length(z));
    z = z + randVec * noiseFactorVec(3);
    
    % Create the current row of pulses  
    scans(:,:,n) = makePulsOn(x0,y0,z0,x,y,z,tauP,fs,fc,c,n,nY);
end 
end  % formPulses

function [yOut] = makePulsOn(x0,y0,z0,x,y,z,tauP,fs,f0,c, curRow, numRows)
% x0,y0,z0 - Lx1 list of target positions
% x,y,z - PxQ list of sensor co-ordinates (row per row)
% t,tauP - PxQ list of pulse start times and pulse durations
% prf - pulse PRF (assume %50 duty up/down)
% fs - sampling frequency
% f0 - center frequency
% bw - Bandwidth
% c - speed of light
% yOut - MxN sample matrix (row per row)
% pOut - Mxn sample matrix of trigger samples

L=length(x0);
[P,Q]=size(x);
d=zeros(P,Q,L);     % distances from point reflectors to positions
for l=1:L
    d(:,:,l)=sqrt((x-x0(l)).^2+(y-y0(l)).^2+(z-z0(l)).^2);
end
tau=d/(c/2);        % roundtrip time from point reflectors to positions
nSamp = ceil(tauP(:,Q)*fs);
nY=max(nSamp);
yOut=zeros(nY, Q);
for p = 1:P
    yOut=sarPulsLine(tauP(p,:),tau(p,:,:),fs,f0,nY,L,curRow, numRows);
end
end 

function [yLine] = sarPulsLine(tauP,tau,fs,f0,nY,numTargets, curRow, numRows)

w0 = 2*pi*f0;
fDemod = fs/4;
Pt = 5/(2 * fs);
numPulses = length(tauP);
yLine = zeros(nY,numPulses); 
deltaT = nY/fs;
step = 1/fs;
tOffset = (0:step:(deltaT - step))';
fprintf('Generating Pulses: Row %d of %d\n', curRow, int16(numRows));
for n = 1:numPulses
    for k = 1:numTargets
        
        yLine(:, n) = yLine(:, n) + (exp(1i*w0.*(tOffset-tau(1,n,k)))) .*...
                                     hannWindow(tau(1,n,k), tau(1,n,k) + ...
                                     10*Pt, tOffset) .* (1/(tau(1,n,k)*(3e8))^4);
    end
    yLine(:, n) = yLine(:, n) .* exp(-1i*2*pi*fDemod.*tOffset);
end
end

function [rectanglePulse] = piFunction(tStart, tStop, vector)
% Creates a rectangular pulse over the given vector
% tStart and tStop are the start and stop times of the pulse

numSamp = length(vector);
tempPulse = zeros(1, numSamp)';
counter = 0;

for i = 1:numSamp
    if vector(i) > tStart && vector(i) < tStop
        counter = counter + 1;
        tempPulse(i) = tempPulse(i) + 1;
    end 
end 
rectanglePulse = tempPulse;
end

function [windowedPulse] = hannWindow(tStart, tStop, vector)

% applies a Window over the given vector, 
% where tStart and tStop are the boundaries of the desired window
% 
% in this case, the window is a Hann Window

numSamp = length(vector);
tempPulse = zeros(numSamp, 1);
counter = 0;
startIdx = [];  foundStart = false;
stopIdx = [];    foundStop = false;
counter = 0;
i = 1;

% 1) locate the start and stop indicies 
% 2) determine the number of pulses in the range
while i < numSamp
    if vector(i) < tStart
        i = i + 1;
    elseif ~foundStart
        startIdx = i;
        foundStart = true;
        i = i + 1;
    elseif vector(i) > tStart && vector(i) < tStop
        i = i + 1;
        counter = counter + 1;
    elseif ~foundStop 
        stopIdx = i;
        foundStop = true;
        i = i + 1;
    else
        break
    end

end

% apply the Hann Window
window = .5 + .5*cos(2*pi*((1:counter).'/(counter+1) - .5));
i = 1;
for j = startIdx:stopIdx - 2
    tempPulse(j) = window(i);
    i = i + 1;
end
windowedPulse = tempPulse;
end 

% -----------------------------------------------------------------------------
% Format raw radar pulse returns from the formPulses function such that
% they are compatible with SAR_3D 
% @param scans [Array]
% @return rawCollect [Array]
% -----------------------------------------------------------------------------
function [rawCollect] = formatPulseData(scans)

C_mps = 299792458;          % (m/s) Speed of light
fLow  = 3100e6;             % (Hz) Waveform operating band
fHigh = 5300e6;             % (Hz) Waveform operating band
fc    = (fLow + fHigh) / 2; % (Hz) Waveform center frequency

fprintf('Formatting pulse data...\n');

numScans = length(scans(1,:,1));
numRows = length(scans(1,1,:));
rawCollect = cell(1,numScans);

delta = C_mps / (4 * fc); % pulses are taken ideally at lambda / 4 apart

for j = 1:numRows
  for i = 1:numScans
    %% Set the position of each pulse within the aperture 
    rawCollect{j,i}.xLoc_m                = (delta*(i-1));
    rawCollect{j,i}.yLoc_m                = 0;
    rawCollect{j,i}.zLoc_m                = (delta*(j-1));
    rawCollect{j,i}.scan                  = real(scans(:,i,j));
    rawCollect{j,i}.scanResPs             = 600;
  end
end

end

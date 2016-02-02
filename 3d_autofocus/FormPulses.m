function [yOut] = FormPulses(x0, y0, z0, nX, nY, addNoise)

% ----------------------------------------------------------------------- %
% Form Pulses
% 
% With given target locations and pulses in the aperture, this function
% will generate artificial radar pulses of the given scene. Designed to
% simulate the PulsOn p410 radar unit as accurately as possible. This
% additionally saves the data to RawImageData.mat.
%
% Inputs:
%   x0, y0, z0 - [ X Y Z ] coordinates of each radar target
%   nX         - number of pulses per row of the aperture
%   nY         - number of rows in the aperture
%   addNoise   - boolean; if true will add random noise in the aperture
%                locations. This noise is typical of the Piksi RTK GPS
%                unit.
%
% ----------------------------------------------------------------------- %

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
     
    if addNoise                 % noise is a normal curve in each dimension
                                % and is based on the expected error from 
       sigmaY = 0.0051;         % the RTK GPS unit
       sigmaZ = 0.0071;
       meanY  = -0.009;
       meanZ  = -0.0083;
       meanX  = 0;              % Further testing needs to be completed to
       sigmaX = 0.0051;         % determine the meanX and sigmaX...
                                % X position is a function of time;
                                % GPS data needs to be correlated with
                                % video frames
                                % sigmaX assumed to be equal to sigmaY
                                % because SwiftNav claims error
                                % distributions are equal in the xy plane
       
       % Add random noise into the aperture positions;
       % Noise was experimentally found to be a normal distrubution curve
       randVec = normrnd(meanZ,sigmaZ,1,length(z));
       z = z + randVec;
       randVec = normrnd(meanY,sigmaY,1,length(y));
       y = y + randVec;
       randVec = normrnd(meanX,sigmaX,1,length(x));
       x = x + randVec;
    end
    
    % Create the current row of pulses  
    yOut(:,:,n) = makePulsOn(x0,y0,z0,x,y,z,tauP,fs,fc,c,n,nY);
    
end 
end  % FormPulses

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

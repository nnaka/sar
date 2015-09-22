% Test file for PulsOn 
function [] = PulsOnTest(sigma)
x0=kron((-20:20:20)',ones(3,1));     % these vectors define the locations of the set of point source scatterers that we model in the scenario nid
y0=kron(ones(3,1),(-20:20:20)');
z0=[0;30;-30;30;0;-30;30;-30;0];
% z0 = zeros(9,1);
%x0 = [-30,-30,-30,-30,-30,-20,-10,-10,-10,-10,-10,10,20,30,20,20,20,10,20,30];
%y0 = [10,5,0,-5,-10,0,10,5,0,-5,-10,10,10,10,5,0,-5,-10,-10,-10];
%z0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
% x0 = 20; y0 = 0; z0 = 0;
addNoise = true;

    maxRange = 1e3;           % max radar range in meters
    c = 299792458;                    % speed of light
    f0 = 4.3e9;                 % radar center frequency
    fs = 1.25e9;                % radar sample rate
    
    droneVel = 1;               % drone speed in m/s
    lambda = c/f0;    

nX = 40;         % width of aperture   
nY = 1;          % Number of rows
yCent = -60;
xCent = 0; xStep = lambda/4;
zCent = 0; zStep = lambda/4;
for n = 1:nY
    x=xCent+xStep*(-(nX-1)/2:1:(nX-1)/2);  % radar positions
    y=yCent*ones(size(x));
    z=zCent+zStep*(n-(nY-1)/2)*ones(size(x));
    if addNoise
       randVec = normrnd(0,sigma,1,length(z));
        z = z + .0108*randVec;
       randVec = normrnd(0,sigma,1,length(y));
       y = y + .0051*randVec;
       randVec = normrnd(0,sigma,1,length(x));
        x = x + .0051*randVec;
    end
    % Assumptions: 
    %
    % Drone elevation: 100m  
    % maximum radar viewing range: 100km
    %            --> listening time of 1us 
    % drone velocity: 1m/s
    %            --> transmit pulse every 0.03s
           % pulse wavelength
    prf = droneVel * 2 / lambda; % pulse repitition frequency
%     t = (1:size(z,2)) * (1/prf);  
    
    scaling = (maxRange*2)/c;
    tauP = ones(nX) * scaling; % listening time

    
    bw = 200e6;                 % pulse bandwidth
    
    yOut = makePulsOn(x0,y0,z0,x,y,z,tauP,fs,prf,f0,c);
    %fid = fopen('PulsOnTestData.txt','w');
    %fprintf(fid, '%bu', yOut);
    %dlmwrite('PulsOnTestData.txt', yOut, ' ');
    
    
    save('PulsOnTestData.mat', 'yOut');
    %fclose(fid);
    %createPlot(yOut);
%     time = (0:length(yOut)-1)/fs;
%     plot(time, yOut)
    
end 

end 

function [] = createPlot(yOut)

numSamples = length(yOut(1,:));
zeroData = zeros(length(yOut(:,1)/8), 1);
yPlot = [];
for i = 1:numSamples
    yPlot = [yPlot, yOut(:,i)'];
   % yPlot = [yPlot, zeroData'];
end 
time = (0:(length(yPlot)-1));
plot(time, real(yPlot));
end

function [yOut] = makePulsOn(x0,y0,z0,x,y,z,tauP,fs,prf,f0,c)
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
    yOut=sarPulsLine(tauP(p,:),tau(p,:,:),fs,prf,f0,nY,L);
end
end 

function [yLine] = sarPulsLine(tauP,tau,fs,prf,f0,nY,numTargets)

w0 = 2*pi*f0;
fDemod = fs/4;
Pt = 5/(2 * fs);
numPulses = length(tauP);
yLine = zeros(nY,numPulses); 
pulse = 1/prf;
deltaT = nY/fs;
step = 1/fs;
tOffset = (0:step:(deltaT - step))';
hW = waitbar(0, 'SAR PulsOn Line');
for n = 1:numPulses
    envelope = zeros(nY,1);
    for k = 1:numTargets
%         yLine(:, n) = yLine(:, n) + (exp(1i*w0.*(tOffset-tau(1,n,k)))...
%                       .*rectangularPulse(0, max(tOffset), max(tauP)));
    
        yLine(:, n) = yLine(:, n) + (exp(1i*w0.*(tOffset-tau(1,n,k)))) .*...
                                         hannWindow(tau(1,n,k), tau(1,n,k) + ...
                                         10*Pt, tOffset) .* (1/(tau(1,n,k)*(3e8))^4);
    end
    yLine(:, n) = yLine(:, n) .* exp(-1i*2*pi*fDemod.*tOffset);
    waitbar(n/numPulses, hW);
end
close(hW);
end

function [rectanglePulse] = piFunction(tStart, tStop, vector)
% Creates a rectangular pulse over the given vector
% tStart and tStop are the start and stop times of the pulse

tempPulse = zeros(1, length(vector))';

% startIndex
for i = 1:length(vector)
    if vector(i) > tStart && vector(i) < tStop
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

















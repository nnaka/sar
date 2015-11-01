x0=kron((-20:20:20)',ones(3,1));     % these vectors define the locations of the set of point source scatterers that we model in the scenario nid
y0=kron(ones(3,1),(-20:20:20)');
z0=[0;30;-30;30;0;-30;30;-30;0];
% x0=0;
% y0=0;
% z0=0;
nX=41;                               % we will model a square grid of radar postions in the swept aperture  
nY=11;                               % this code uses a square grid but any assortement of points can be used
yCent=-80;                           % this is the place to experiment with aperture geometry
xCent=0;xStep=.05;                   % we'll sweep out a an aperture at x,y,z=[0,-80,0] and looking at targets
zCent=0;zStep=.05;                   % centered around [0 0 0] - play with geometry and see what happens
for n=1:nY
    fprintf('pass #%d\n',n);         % create a separate file to contain the signal for each line of antenna positions
    fileName=sprintf('test%03d.wav',n);    %file name
    x=xCent+xStep*(-(nX-1)/2:1:(nX-1)/2);  % radar positions
    y=yCent*ones(size(x));
    z=zCent+zStep*(n-(nY-1)/2)*ones(size(x));
    t=(1:size(z,2))*2; % +.05*randn(1,size(z,2));      % time when the pulses assocaited with each position start (assuming 2 seconds per positions with 1 sec of transmit time)
    tauP=ones(size(t));
    fs=44100;                                          % rate at which our signal is samples (44.1KHz is a PC audio board sampling speed)
    prf=5;                                             % pulse repetition rate 5Hz
    f0=2.4e9;                                          % radar's center frequency
    bw=200e6;                                          % pulse bandwidth
    c=3e8;                                             % speed of light
    
    [yOut,pOut]=makeSAR(x0,y0,z0,x,y,z,t,tauP,fs,prf,f0,bw,c);   % create pulses (yOut) and a synch signal (pOut) associated with returns from a line of positions
    tAx=(0:length(yOut)-1)/fs;
    plot(tAx,pOut,tAx,yOut)                                      % plot pulses vs time
    G=2^12;
    wavwrite(int16(G*[-pOut(:) -yOut(:)]),fs,fileName)           % write the file - these files will be used to form the SAR image - we will see the targets defined by x0,y0,z0 when we process these files to an image
end
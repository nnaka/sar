f=4.3e9;      % receiver center frequency (hz) (pulseON value)
Nx=101;       % number of positions to cross the aperture in azimuth
Ny=101;       % number of positions to cross the aperture in elevation
Bw=1.4e9;     % pulse bandwidth (hz) (pulseOn Value)
c=3e8;        % speed of light (m/s)
lambda=c/f;   % radar wavelength
R=20;         % range (m) at which to form image 
Ax=(Nx-1)*lambda/2;            
Ay=(Ny-1)*lambda/2;
dx=R*lambda/(2*Ax);        % 3dB-3dB beam size - double for null-null beam size
dy=R*lambda/(2*Ay);        % 3dB-3dB beam size - double for null-null beam size
dr=c/(2*Bw);               % range resolution
maxPRF=c/(2*R);            % maximum PRF for range extent
fprintf('SAR Aperture %7.2f x %7.2f meters (az x el)\n',Ax,Ay);
fprintf('Resolution %7.4f x %7.4f x %7.4f meters(Az x El x Range)\n',dx,dy,dr);
fprintf('maximum PRF for desired range swath: %7.2f KHz\n',maxPRF/1e3)
fprintf('range to swath: %7.2f meters\n',R)

% makeSarImage - create a set of wavefiles representing returns from
% a set of point scatterers toward a syntheticly swept set of sar positions
% fileRoot - string used to generate files.  The files will be named
% fileRoot0001.wav-fileRoot00My.wav
% targX,targY,targZ - Lx1 x,y,z locations of point scatterers (m)
% dx,dy,dz,Nx,My - used to describe swept aperture (m)
%   a line of nX positions as the antenna is moved in x will be dx meters
%   apart and centered at x=0
%   A total of nY lines will be generated and these will be offset from
%   each other by dy, centered on y=0
%   a z displacement between lines can be specified, the lines will be
%   centered on z=0
% f0 - center frequency (Hz)
% fs - sampling frequency (Hz)
% prf - pulse repeat frequency (%50 duty is assumed) (Hz)
% t - nX x nY list of cpi start times (s)
% Tp - nX x nY list of cpi durrations (s)
%  t must grow a in each column and t(i,j)+Tp(i,j)> t(i+1,j)
% bw - bandwidth of LFM chirp (Hz)
% c - speed of light


function makeSarImage(fileRoot,targX,targY,targZ,dx,dy,dz,nX,nY,t,Tp,f0,fs,prf,bw,c)

xArray=repmat(linspace(-(nX-1)/2*dx,(nX-1)/2*dx,nX),nY,1);
yArray=repmat(linspace(-(nY-1)/2*dy,(nY-1)/2*dy)',1,nX);
zArray=repmat(linspace(-(nY-1)/2*dz,(nY-1)/2*dz)',1,nX);

for n=1:nY
    fprintf('pass #%d\n',n);
    fileName=[fileRoot sprintf('%04d.wav',n)];
    x=xArray(n,:);
    y=yArray(n,:);
    z=zArray(n,:);
    [yOut,pOut]=makeSAR(targX,targY,targZ,x,y,z,t(:,n)',Tp(:,n)',fs,prf,f0,bw,c);
    tAx=(0:length(yOut)-1)/fs;
    plot(tAx,pOut,tAx,yOut)
    G=2^12;
    wavwrite(int16(G*[-pOut(:) -yOut(:)]),fs,fileName)
end
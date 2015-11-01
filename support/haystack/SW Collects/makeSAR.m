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
function [yOut,pOut]=makeSAR(x0,y0,z0,x,y,z,t,tauP,fs,prf,f0,bw,c)
L=length(x0);
[P,Q]=size(x);
d=zeros(P,Q,L);     % distances from point reflectors to positions
for l=1:L
    d(:,:,l)=sqrt((x-x0(l)).^2+(y-y0(l)).^2+(z-z0(l)).^2);
end
tau=d/(c/2);        % roundtrip time from point reflectors to positions
nSamp=ceil((t(:,Q)+tauP(:,Q)+1)*fs);         % Px1 list of samples produced by each row
nY=max(nSamp);
yOut=zeros(P,nY);
pOut=yOut;
for p=1:P
    [yOut(p,:),pOut(p,:)]=sarLine(t(p,:),tauP(p,:),tau(p,:,:),fs,prf,f0,bw,nY);
end


x0=kron((-30:20:30)',ones(4,1));
y0=kron(ones(4,1),(-30:20:30)');
z0=zeros(size(y0));
% x0=[0];
% y0=[0];
% z0=[0];
x=.05*(-20:20);
y=-80*ones(size(x));
z=zeros(size(x));
t=(1:size(z,2))*2;
tauP=ones(size(t));
fs=44100;
prf=4;
f0=2.4e9;
bw=200e6;
c=3e8;

[yOut,pOut]=makeSAR(x0,y0,z0,x,y,z,t,tauP,fs,prf,f0,bw,c);
tAx=(0:length(yOut)-1)/fs;
plot(tAx,pOut,tAx,yOut)
G=2^12;
wavwrite(int16(G*[pOut(:) yOut(:)]),fs,'testOut.wav')

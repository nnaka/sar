t=0:1e-3:1;
f=40;
y=sin(2*pi*f*t).*rectangularPulse(0,.1,t);
plot(t,y,'k')
boldify
set(get(1,'children'),'visible','off')
set(get(1,'children'),'box','off','color','none')
thb=.5;
thd=.02;
ttr=.7;
pl=.1;
for n=1:5
    yt=sin(2*pi*f*(t-ttr)).*rectangularPulse(0,pl,t-ttr);
    yh=sin(2*pi*f*(t-thb+n*thd)).*rectangularPulse(0,pl,t-thb+n*thd);
    idx=find(t>=thb-n*thd & t<=thb-n*thd+pl);
    plot(t,yh+yt,'r',t(idx),yh(idx),'b')
    boldify
    set(get(1,'children'),'visible','off')
    set(get(1,'children'),'box','off','color','none')
    fname=sprintf('ret%d.png',n);
    print(1,fname,'-dpng')
    pause(.1)
end
%%
nR=256;
nP=64;
pT=1000;
pC=1000;
pN=1;
xn=pC*ones(nP,1)*randn(1,nR)+pN*randn(nP,nR);
ax=0:nR-1;
f=.2;
w=kaiser(nP,10);
for n=1:nP
    ph=0+n*.5;
    xn(n,:)=xn(n,:)+w(n)*pT*exp(j*2*pi*f*(ax-ph)).*rectangularPulse(18,20,ax);
    ph=0+n*1.5;
    xn(n,:)=xn(n,:)+w(n)*3*pT*exp(j*2*pi*f*(ax-ph)).*rectangularPulse(118,125,ax);
end
imagesc(db20(fftshift(fft(xn),1)))
axis('xy')
cax=caxis;
caxis([cax(2)-100 cax(2)])
boldify
    set(get(1,'children'),'visible','off')
    set(get(1,'children'),'box','off','color','none')
print(1,'MTI_dop_range','-dpng')
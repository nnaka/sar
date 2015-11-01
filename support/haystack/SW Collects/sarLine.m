% t - 1xQ list of pulse starts
% tauP - 1xQ list of pulse durations
% tau - 1xQxL list of roudtrip times from L targets to Q locations
% fs - sampling rate
% prf - pulse repeat frequency (assume %50 duty)
% f0 - center frequecy
% bw - bandwidth
% nY - length of yLine and pLine
% yLine - output of pulse returns
% pLine - output of sync pulses

function [yLine,pLine] = sarLine(t,tauP,tau,fs,prf,f0,bw,nY)
D=4;pVal=1;nVal=-1;
w0=2*pi*f0;
axD=-D:D;
axT=1:nY;
yLine=zeros(1,nY);
pLine=zeros(1,nY);
lT=length(t);
pulse=1/prf;
halfPulse=.5*pulse;
quarterPulse=.25*pulse;
alpha=pi*bw/halfPulse;
hW=waitbar(0,'SAR Line');
for n=1:lT
    waitbar(n/lT,hW);
    teP=t(n)+(0:pulse:tauP);
    tauN=squeeze(tau(1,n,:));
    for nn=1:length(teP)
        axP=find((axT-1)/fs>teP(nn) & (axT-1)/fs<=teP(nn)+halfPulse);
        axN=find((axT-1)/fs>teP(nn)+halfPulse & (axT-1)/fs<=teP(nn)+pulse);
        pLine(axP)=pVal;
        pLine(axN)=nVal;
        axISin=axP(1)+axD;
        offset=(axP(1)-1)-teP(nn)*fs;
        if nn==1
            pLine(axISin)=pVal/2+pVal*sinint(pi*(axD+offset))/pi;
        else
            pLine(axISin)=(pVal+nVal)/2+(pVal-nVal)*sinint(pi*(axD+offset))/pi;
        end
        yLine(axP)=sum(((.5e-6./tauN(:)).^4)*ones(size(axP)).*...
                   cos(-w0*tauN(:)*ones(size(axP))...
                       -2*alpha*tauN(:)*((axP-1)/fs-(teP(nn)+quarterPulse))...
                       +alpha*tauN(:).^2*ones(size(axP))),1);
        axISin=axN(1)+axD;
        offset=(axN(1)-1)-(teP(nn)+halfPulse)*fs;
        if nn==length(teP)
            pLine(axISin)=nVal/2+nVal*sinint(pi*(axD+offset))/pi;
        else
            pLine(axISin)=(pVal+nVal)/2-(pVal-nVal)*sinint(pi*(axD+offset))/pi;
        end
        yLine(axN)=sum(((.5e-6./tauN(:)).^4)*ones(size(axN)).*...
                   cos(-w0*tauN(:)*ones(size(axN))...
                       +2*alpha*tauN(:)*((axN-1)/fs-(teP(nn)+halfPulse+quarterPulse))...
                       +alpha*tauN(:).^2*ones(size(axN))),1);
    end
end
close(hW)
end


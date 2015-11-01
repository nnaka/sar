function plot_multi_double_scn(srl,Nrqst)
% PLOT_MULTI_DOUBLE_SCN Function to generate and plot multiple double scans.


R1 = 1;  % m
R2 = 10;  % m
[T1,T2,R1,R2,Rbin,Nbin] = rdr_scn_setup(R1,R2);
%R1
%R2

Gtx = 63;
PII = 7;
chng_cfg(srl,[T1 T2],Gtx,PII)

%view_cfg(s)

SCNmsgNbin = 350;  % number of bins in each message (see API)
USBpfxNbyt = 4;
CFRMmsgNbyt = 8;
SCNmsgNbyt = 1452;

Nscn = 2;

Nmsg = ceil(Nbin/SCNmsgNbin);
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nscn*Nmsg*(USBpfxNbyt + SCNmsgNbyt);

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
hold on
grid on
xlabel('range (m)')
ylabel('amplitude')
ylim(20000*[-1 1])

Hplt1 = plot(Rbin,zeros(1,Nbin),'b.-');
Hplt2 = plot(Rbin,zeros(1,Nbin),'g.-');
Hplt3 = plot(Rbin,zeros(1,Nbin),'r.-');
Hplt4 = plot(Rbin,zeros(1,Nbin),'k.-');

for k = 1:Nrqst
  ctl_rqst(srl,Nscn,12500,k)
  
  Ktry = 0;
  
  while srl.BytesAvailable < totNbyt && Ktry <= 10
    
    Ktry = Ktry + 1;
    
    pause(0.0001)
    
  end
  
  if Ktry <= 10
    
    msg = uint8(fread(srl,srl.BytesAvailable,'uint8'));
    
    Ibyt = 1;
    
    Ibyt = Ibyt + USBpfxNbyt;
    [str,msg_typ,msgID] = parse_msg(msg(Ibyt:Ibyt+CFRMmsgNbyt-1));
    Ibyt = Ibyt + CFRMmsgNbyt;
    
    SCN = int32(zeros(Nscn,Nbin));
    
    for m = 1:Nscn
      for n = 1:Nmsg
        Ibyt = Ibyt + USBpfxNbyt;
        [str,msg_typ,msgID] = parse_msg(msg(Ibyt:Ibyt+SCNmsgNbyt-1));
        Ibyt = Ibyt + SCNmsgNbyt;
        
        Ibin = SCNmsgNbin*(n - 1) + 1;
        SCN(m,Ibin:Ibin+str.messageSamples-1) = str.scanData(1:str.messageSamples);
      end
    end
    
  else
    fprintf('Scan data not returned.')
    
  end
  
  set(Hplt1,'YData',SCN(1,:))
  set(Hplt2,'YData',SCN(2,:))
  set(Hplt3,'YData',diff(SCN,1,1))
  set(Hplt4,'YData',envelope(double(diff(SCN,1,1))))
end

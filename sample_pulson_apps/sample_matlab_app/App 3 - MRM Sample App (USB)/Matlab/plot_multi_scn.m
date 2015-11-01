function plot_multi_scn(srl,Nrqst)
% PLOT_MULTI_SCN Function to generate and plot multiple scans.


R1 = 2;  % m
R2 = 12;  % m
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

Nscn = 1;

Nmsg = ceil(Nbin/SCNmsgNbin);
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nscn*Nmsg*(USBpfxNbyt + SCNmsgNbyt);

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
hold on
grid on
xlabel('range (m)')
ylabel('amplitude')
ylim(20000*[-1 1])

SCN = int32(zeros(1,Nbin));

Hplt1 = plot(Rbin,SCN,'b.-');
Hplt2 = plot(Rbin,SCN,'r.-');
Hplt3 = plot(Rbin,SCN,'k.-');

for k = 1:Nrqst
  SCNprev = SCN;
  
  msgID = k;
  ctl_rqst(srl,Nscn,0,msgID)
  
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
    
    SCN = [];
    
    for n = 1:Nmsg
      Ibyt = Ibyt + USBpfxNbyt;
      [str,msg_typ,msgID] = parse_msg(msg(Ibyt:Ibyt+SCNmsgNbyt-1));
      Ibyt = Ibyt + SCNmsgNbyt;
      
      SCN = [SCN str.scanData(1:str.messageSamples)];
    end
    
  else
    fprintf('Scan data not returned.')
    
  end
  
  set(Hplt1,'YData',SCN)
  set(Hplt2,'YData',SCN - SCNprev)
  set(Hplt3,'YData',envelope(double(SCN - SCNprev)))
end

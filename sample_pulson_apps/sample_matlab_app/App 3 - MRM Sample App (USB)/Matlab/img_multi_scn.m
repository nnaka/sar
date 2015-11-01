function img_multi_scn(srl,Nrqst)
% IMG_MULTI_SCN Function to generate multiple scans and create envelope image.


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

Nscn = 1;

Nmsg = ceil(Nbin/SCNmsgNbin);
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nscn*Nmsg*(USBpfxNbyt + SCNmsgNbyt);

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
SCNenv = ones(Nrqst,Nbin);
Himg = image(Rbin,1:Nscn,SCNenv);
hold on
grid on
xlabel('range (m)')
ylabel('scan')

SCNenv = zeros(Nrqst,Nbin);

SCN = int32(zeros(1,Nbin));

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
  
  if k > 1
    SCNenv(k,:) = envelope(double(SCN - SCNprev));
    SCNenv(k,:) = min(round(63*SCNenv(k,:)/10e3) + 1,64);
    % NOTE: For k == 1, the previous scan is all zeros, so the difference
    % is just the current scan, which is very large and obscures the other
    % scans in the image.
  end
  
  set(Himg,'CData',SCNenv)
  
  Idet = SCNenv(k,:) >= 10;
  if sum(Idet) >= 5
    plot(Rbin(find(Idet,1,'first')),k,'Color','w','Marker','.','MarkerSize',12,'LineStyle','none')
  end
end

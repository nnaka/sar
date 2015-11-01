function img_scn_fft(srl,Nrqst,Nfft)
% IMG_SCN_FFT Function to generate and create images of FFTs for multiple scans.


R1 = 1;  % m
R2 = 5;  % m
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

Nmsg = ceil(Nbin/SCNmsgNbin);
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nfft*Nmsg*(USBpfxNbyt + SCNmsgNbyt);

Fpls = 80;  % Hz
Tpls = 1e6/Fpls;  % us
Ffft = (-Nfft/2:Nfft/2-1)*Fpls/Nfft;

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')

Himg = image(Ffft,Rbin,ones(Nbin,Nfft));

hold on
grid on
xlabel('Doppler (Hz)')
ylabel('range (m)')
set(gca,'YDir','normal')

for k = 1:Nrqst
  ctl_rqst(srl,Nfft,Tpls,1)
  
  Ktry = 0;
  
  while srl.BytesAvailable < totNbyt && Ktry <= 100
    
    Ktry = Ktry + 1;
    
    pause(0.0001)
    
  end
  
  if Ktry <= 100
    
    msg = uint8(fread(srl,srl.BytesAvailable,'uint8'));
    
    Ibyt = 1;
    
    Ibyt = Ibyt + USBpfxNbyt;
    [str,msg_typ,msgID] = parse_msg(msg(Ibyt:Ibyt+CFRMmsgNbyt-1));
    Ibyt = Ibyt + CFRMmsgNbyt;
    
    SCN = int32(zeros(Nfft,Nbin));
    
    for m = 1:Nfft
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
  
  IQ = hilbert(double(SCN'));
  
  fftSCN = fft(IQ,Nfft,2);
  fftSCN = abs(fftSCN);
  fftSCN(:,1) = 0;
  fftSCN = fftshift(fftSCN,2);
  fftSCN = min(round(63*fftSCN/1.6e5) + 1,64);
  
  set(Himg,'CData',fftSCN)
end

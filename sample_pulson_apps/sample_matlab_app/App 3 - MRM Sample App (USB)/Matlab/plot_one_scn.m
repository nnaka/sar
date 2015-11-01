function plot_one_scn(srl)
% PLOT_ONE_SCN Function to generate and plot one scan.


R1 = 2;  % m
R2 = 12;  % m
[T1,T2,R1,R2,Rbin,Nbin] = rdr_scn_setup(R1,R2);
R1
R2

Gtx = 63;
PII = 7;
chng_cfg(srl,[T1 T2],Gtx,PII)

view_cfg(srl)

SCNmsgNbin = 350;  % number of bins in each message (see API)
USBpfxNbyt = 4;
CFRMmsgNbyt = 8;
SCNmsgNbyt = 1452;

Nscn = 1;

Nmsg = ceil(Nbin/SCNmsgNbin)
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nscn*Nmsg*(USBpfxNbyt + SCNmsgNbyt)

msgID = 1;
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
    
    msg_typ
    str
    SCN = [SCN str.scanData(1:str.messageSamples)];
  end
  
else
  fprintf('Scan data not returned.')
  
end

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
hold on
grid on
xlabel('range (m)')
ylabel('amplitude')

plot(Rbin,SCN,'b.-')

%{
Here is a list of messages and bytes that result from the control request:

     NUM       FIRST        LAST
   BYTES        BYTE        BYTE
                                    ----- MRM_CONTROL_CONFIRM -----
       4           1           4    USB prefix (0xA5A5 & number of bytes)
       8           5          12    message data

                                    ----- MRM_SCAN_INFO -----
       4          13          16    USB prefix (0xA5A5 & number of bytes)
      52          17          68    message header data
    1400          69        1468    message scan data (message 1/4) [350 bins]
       4        1469        1472    USB prefix (0xA5A5 & number of bytes)
      52        1473        1524    message header data
    1400        1525        2924    message scan data (message 2/4) [350 bins]
       4        2925        2928    USB prefix (0xA5A5 & number of bytes)
      52        2929        2980    message header data
    1400        2981        4380    message scan data (message 3/4) [350 bins]
       4        4381        4384    USB prefix (0xA5A5 & number of bytes)
      52        4385        4436    message header data
    1400        4437        5836    message scan data (message 4/4) [350 bins]

Note that the last MRM_SCAN_INFO message only contains 6 bins with valid
scan values.

Note that the very last byte value (5836), which is the total number of
bytes, is the BytesAvailable number that triggers fread.
%}

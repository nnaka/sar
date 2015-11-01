function plot_one_double_scn(srl)
% PLOT_ONE_DOUBLE_SCN Function to generate and plot one double scan.


R1 = 1;  % m
R2 = 10;  % m
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

Nscn = 2;

Nmsg = ceil(Nbin/SCNmsgNbin)
totNbyt = USBpfxNbyt + CFRMmsgNbyt + Nscn*Nmsg*(USBpfxNbyt + SCNmsgNbyt)

ctl_rqst(srl,Nscn,12500,1)

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
      
      msg_typ
      str
      Ibin = SCNmsgNbin*(n - 1) + 1;
      SCN(m,Ibin:Ibin+str.messageSamples-1) = str.scanData(1:str.messageSamples);
    end
  end
  
else
  fprintf('Scan data not returned.')
  
end

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Color','w')
hold on
grid on
xlabel('range (m)')
ylabel('amplitude')

plot(Rbin,SCN(1,:),'b.-')
plot(Rbin,SCN(2,:),'g.-')
plot(Rbin,diff(SCN,1,1),'r.-')

%{
Here is a list of messages and bytes that result from the control request:

     NUM       FIRST        LAST
   BYTES        BYTE        BYTE
                                    ----- MRM_CONTROL_CONFIRM -----
       4           1           4    USB prefix (0xA5A5 & number of bytes)
       8           5          12    message data

                                    ----- MRM_SCAN_INFO ----- (set of messages for first requested scan)
       4          13          16    USB prefix (0xA5A5 & number of bytes)
      52          17          68    message header data
    1400          69        1468    message scan data (message 1/3) [350 bins]
       4        1469        1472    USB prefix (0xA5A5 & number of bytes)
      52        1473        1524    message header data
    1400        1525        2924    message scan data (message 2/3) [350 bins]
       4        2925        2928    USB prefix (0xA5A5 & number of bytes)
      52        2929        2980    message header data
    1400        2981        4380    message scan data (message 3/3) [350 bins]

                                    ----- MRM_SCAN_INFO ----- (set of messages for second requested scan)
       4        4381        4384    USB prefix (0xA5A5 & number of bytes)
      52        4385        4436    message header data
    1400        4437        5836    message scan data (message 1/3) [350 bins]
       4        5837        5840    USB prefix (0xA5A5 & number of bytes)
      52        5841        5892    message header data
     160        5893        6052    message scan data (message 2/3) [40 bins]
    1240        6053        7292    unused scan data (in message 5/3)
       4        7293        7296    USB prefix (0xA5A5 & number of bytes)
      52        7297        7348    message header data
    1400        7349        8748    message scan data (message 3/3) [350 bins]

Note that the last MRM_SCAN_INFO message in each set only contains 260 bins
with valid scan values.

Note that the very last byte value (8748), which is the total number of
bytes, is the BytesAvailable number that triggers fread.
%}

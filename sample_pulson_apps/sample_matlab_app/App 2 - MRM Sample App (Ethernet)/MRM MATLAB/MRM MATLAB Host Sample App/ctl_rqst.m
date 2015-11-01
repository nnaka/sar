function ctl_rqst(IPaddr,msgID,ctl)
% CTL_RQST Control request function.


import java.io.*
import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.InetAddress

port = 21210;
sckt = sckt_mgr('get');

MSG_TYPE = uint16(hex2dec('1003')); 
MSG_ID = uint16(msgID);
DAT = str2dat(ctl);
CTL_RQST = [typecast(swapbytes([MSG_TYPE MSG_ID]),'int8') DAT];

try
  addr = InetAddress.getByName(IPaddr);
  pckt = DatagramPacket(CTL_RQST,length(CTL_RQST),addr,port);
  sckt.send(pckt);
  
catch err
  fprintf('%s\n',err.message)
  error('Failed to send packet.');
  
end

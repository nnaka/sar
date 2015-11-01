function get_cfg_rqst(IPaddr,msgID)
% GET_CFG_RQST Get configuration request function.


import java.io.*
import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.InetAddress

port = 21210;
sckt = sckt_mgr('get');

MSG_TYPE = uint16(hex2dec('1002')); 
MSG_ID = uint16(msgID);
GET_CFG_RQST = typecast(swapbytes([MSG_TYPE MSG_ID]),'int8');

try
  addr = InetAddress.getByName(IPaddr);
  pckt = DatagramPacket(GET_CFG_RQST,length(GET_CFG_RQST),addr,port);
  sckt.send(pckt);
  
catch err
  fprintf('%s\n',err.message)
  error('Failed to send packet.\n');
  
end

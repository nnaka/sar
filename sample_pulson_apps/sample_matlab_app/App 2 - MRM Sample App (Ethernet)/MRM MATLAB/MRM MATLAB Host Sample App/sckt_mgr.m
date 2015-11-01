function sckt = sckt_mgr(op)
% SCKT_MGR Socket manager function.


import java.io.*
import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.InetAddress

persistent SCKT

switch op
  case 'open'
    SCKT = DatagramSocket;
    
  case 'get'
    
  case 'close'
    SCKT.close
    SCKT = [];
    
  otherwise
    error('Invalid operation.')
    
end

sckt = SCKT;

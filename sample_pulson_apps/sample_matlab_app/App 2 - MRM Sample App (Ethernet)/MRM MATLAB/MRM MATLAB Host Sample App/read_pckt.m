function [msg,msg_typ,msgID,IPaddr] = read_pckt
% READ_PCKT Read packet function.


import java.io.*
import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.InetAddress

sckt = sckt_mgr('get');

TIME_OUT = 400;  % ms - if the PC is slow this might need to be increased
PCKT_LEN = 1500;  % max bytes in a UDP packet

try
  sckt.setSoTimeout(TIME_OUT);
  pckt = DatagramPacket(zeros(1,PCKT_LEN,'int8'),PCKT_LEN);        

  sckt.receive(pckt);
  msg = pckt.getData;

  addr = pckt.getAddress;
  IPaddr = char(addr.getHostAddress);
  
  msg = msg(1:pckt.getLength);     
  msg_typ = dec2hex(typecast([msg(2) msg(1)],'uint16'));
  msgID = typecast([msg(4) msg(3)],'uint16');
  
catch err
  msg = err.message;
  msg_typ = '';
  msgID = [];
  IPaddr = '';
  fprintf('%s\n',err.message)
  error('Timed out attempting to read packet.');
end


% Open a socket if not already open.
sckt = sckt_mgr('get');
if isempty(sckt)
  sckt_mgr('open')
end

% Send a request to get the configuration and then read and parse the
% response packet.
IPaddr = '192.168.1.125';

get_cfg_rqst(IPaddr,1)

[msg,msg_typ,msgID,IPaddr] = read_pckt;

cfg = parse_msg(msg)

% Copy the configuration and change a parameter value. Note that the new
% value must have the correct data type. This could be done automatically
% by getting the value of a parameter, querying for its data type, and then
% setting the new value to be this data type.
CFG = cfg;
CFG.transmitGain = uint8(16);

% Send a request to set the configuration and then read the response
% packet, which is just a confirmation of that the configuration was set.
set_cfg_rqst(IPaddr,2,CFG)

[msg,msg_typ,msgID,IPaddr] = read_pckt;
msg_typ

% Send a request to get the configuration and then read and parse the
% response packet, which will show the changed parameter value.
get_cfg_rqst(IPaddr,3)

[msg,msg_typ,msgID,IPaddr] = read_pckt;

cfg = parse_msg(msg)

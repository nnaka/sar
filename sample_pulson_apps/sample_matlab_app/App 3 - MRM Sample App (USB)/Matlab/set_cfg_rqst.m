function set_cfg_rqst(srl,msgID,cfg)
% SET_CFG_RQST Set configuration request function.
%
% Syntax
% set_cfg_rqst(srl,msgID,cfg)
%
% Input
% srl - serial object
% msgID - integer message ID
%
% Output
% NONE
%
% Usage Notes
% Creates and writes set configuration message.
%
% Copyright © 2014 Time Domain, Huntsville, AL


DAT = str2dat(rmfield(cfg,'timeStamp'));

SYNC_PAT = uint16(hex2dec('A5A5'));
PCKT_LEN = uint16(length(DAT)+4);
MSG_TYPE = uint16(hex2dec('1001')); 
MSG_ID = uint16(msgID);
SET_CFG_RQST = [typecast(swapbytes([SYNC_PAT PCKT_LEN MSG_TYPE MSG_ID]),'int8') DAT];

fwrite(srl,SET_CFG_RQST,'int8')

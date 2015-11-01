function get_cfg_rqst(srl,msgID)
% GET_CFG_RQST Get configuration request function.
%
% Syntax
% get_cfg_rqst(srl,msgID)
%
% Input
% srl - serial object
% msgID - integer message ID
%
% Output
% NONE
%
% Usage Notes
% Creates and writes configuration request message.
%
% Copyright © 2014 Time Domain, Huntsville, AL


SYNC_PAT = uint16(hex2dec('A5A5'));
PCKT_LEN = uint16(4);
MSG_TYPE = uint16(hex2dec('1002')); 
MSG_ID = uint16(msgID);
GET_CFG_RQST = typecast(swapbytes([SYNC_PAT PCKT_LEN MSG_TYPE MSG_ID]),'uint8');

fwrite(srl,GET_CFG_RQST,'uint8');

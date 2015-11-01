function varargout = ctl_rqst(srl,Nscn,dTscn,msgID)
% CTL_RQST Control request function.
%
% Syntax
% ctl_rqst(srl,Nscn,dTscn,msgID)
% CTL_RQST = ctl_rqst(srl,Nscn,dTscn,msgID)
%
% Input
% srl - serial object
% Nscn - number of scans
% dTscn - interval between start of multiple scans (us)
% msgID - integer message ID
%
% Output
% CTL_RQST - control request message bytes array
%
% Usage Notes
% Creates control request message. If no output argument is specified, the
% message is written to the serial port. If the output argment is
% specified, the message bytes are returned as an array and can be written
% manually using FWRITE.
%
% Copyright © 2014 Time Domain, Huntsville, AL


SYNC_PAT = uint16(hex2dec('A5A5'));
PCKT_LEN = uint16(12);
MSG_TYPE = uint16(hex2dec('1003')); 
MSG_ID = uint16(msgID);  % message ID
SCN_CNT = uint16(Nscn);  % number of scans
RSRV = uint16(0);
SCN_INT_TM = uint32(dTscn);  % time in us between scans

CTL_RQST = [typecast(swapbytes([SYNC_PAT PCKT_LEN MSG_TYPE MSG_ID SCN_CNT RSRV]),'uint8') typecast(swapbytes(SCN_INT_TM),'uint8')];

switch nargout
  case 0
    fwrite(srl,CTL_RQST,'uint8')
  case 1
    varargout{1} = CTL_RQST;
end

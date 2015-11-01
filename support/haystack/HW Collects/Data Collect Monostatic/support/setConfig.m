function setConfig(serialObject, msgID, CFG)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Sends packet setting Time Domain P410 configuration.
%
% Inputs:
%   serialObject: Matlab serial object
%   msgID: message ID
%   cfg: P410 configuration structure
% 
% Outputs:
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% Define message parameters
MSG_TYPE = uint16(hex2dec('1001')); 
MSG_ID = uint16(msgID);
DAT = struct2bin(rmfield(CFG,'timeStamp'));

% Construct message (first four bytes are USB header)
SET_CFG_RQST = uint8([hex2dec('A5') hex2dec('A5') 0 4+length(DAT) typecast(swapbytes([MSG_TYPE MSG_ID]),'uint8') DAT]);

% Send message
fwrite(serialObject,SET_CFG_RQST);
function getConfig(serialObject, msgID)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Sends packet requesting Time Domain P410 configuration.
%
% Inputs:
%   serialObject: Matlab serial object
%   msgID: message ID
%
% Outputs:
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% Define message parameters
MSG_TYPE = uint16(hex2dec('1002')); 
MSG_ID = uint16(msgID);

% Construct message (first four bytes are USB header)
GET_CFG_RQST = uint8([hex2dec('A5') hex2dec('A5') 0 4 typecast(swapbytes([MSG_TYPE MSG_ID]),'uint8')]);

% Send message
fwrite(serialObject,GET_CFG_RQST);
  

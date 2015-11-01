function setControl(serialObject,msgID,ctl)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Sends packet controlling Time Domain P410.
%
% Inputs:
%   serialObject: Matlab serial object
%   msgID: message ID
%   ctl: control structure
%
% Outputs:
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% Define message parameters
MSG_TYPE = uint16(hex2dec('1003')); 
MSG_ID = uint16(msgID);

% Construct message (first four bytes are USB header)
DAT = struct2bin(ctl);
CTL_RQST = uint8([hex2dec('A5') hex2dec('A5') 0 4+length(DAT) typecast(swapbytes([MSG_TYPE MSG_ID]),'uint8') DAT]);

% Send message
fwrite(serialObject,CTL_RQST);
function setOpMode(serialObject, msgID, opMode)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Sends packet setting Time Domain P410 operational mode.
%
% Inputs:
%   serialObject: Matlab serial object
%   msgID: message ID
%   opMode: operation mode (1: MRM, 2: RCM, 3: CAT)
%
% Outputs:
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% Define message parameters
MSG_TYPE = uint16(hex2dec('F003')); 
MSG_ID = uint16(msgID);
OP_MODE = uint32(opMode);

% Construct message (first four bytes are USB header)
SET_OP_MODE = uint8([hex2dec('A5') hex2dec('A5') 0 4+4 typecast(swapbytes([MSG_TYPE MSG_ID]),'uint8') typecast(swapbytes(OP_MODE),'uint8')]);

% Send message
fwrite(serialObject,SET_OP_MODE);
  

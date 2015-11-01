function [msg, msg_typ, msgID] = readPacket(serialObject)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Read USB data packet from Time Domain P410 radio.
%
% Inputs:
%   serialObject: Matlab serial object
%
% Outputs:
%   msg: raw message
%   msg_typ: message type
%   msgID: message ID
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% Initialize return variables
msg = [];
msg_typ = [];
msgID = [];

% Turn off timeout warning
warning('off','MATLAB:serial:fread:unsuccessfulRead');

% Try to read from device
rawHeader = fread(serialObject,4,'uint8');

% Check for timeout
if ~isempty(rawHeader)

    % Get packet length
    pcktHeader = uint8(rawHeader);
    pcktLength = double(swapbytes(typecast([pcktHeader(3) pcktHeader(4)],'uint16')));

	% Try to read from device
    rawMessage = fread(serialObject,pcktLength);
    
	% Check for timeout    
    if ~isempty(rawMessage)
        msg = uint8(rawMessage);
        msg_typ = dec2hex(typecast([msg(2) msg(1)],'uint16'));
        msgID = typecast([msg(4) msg(3)],'uint16');
    end

end
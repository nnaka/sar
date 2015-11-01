function activeSerialList = detectActiveRadios()

% MIT IAP 2013: Find a Needle in a Haystack
%
% Runs through available COM ports and determines which are associated with 
% Time Domain P410 radios.  Remember to plug in radios before starting
% Matlab.
%
% Inputs:
%
% Outputs:
%   activeSerialList: cell array containing list of active radios
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

addpath('support');

serialInfo = instrhwinfo('serial');

activeSerialList = {};

fprintf('\nSearching for active P410 radios...\n');

for i=1:length(serialInfo.AvailableSerialPorts)
    
    % Open socket
    s = serial(serialInfo.AvailableSerialPorts(i),'InputBufferSize',15000,'Timeout',0.01);
    
    % Error handling
    try
        fopen(s);
    catch
        continue;
    end
    
    % Query for configuration
    getConfig(s,1);
    
    % Try to read
    [msg, msgType, msgID] = readPacket(s);
    
    if ~isempty(msgID) && strcmp(msgType,'1102') % Active Radio
        activeSerialList{end+1} = serialInfo.AvailableSerialPorts{i};
    else % No response
        continue;
    end
    
    fclose(s);
    
end
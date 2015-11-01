function view_cfg(srl)
% VIEW_CFG Function to view configuration.
%
% Syntax
% view_cfg(srl)
%
% Input
% srl - serial object
%
% Output
% NONE
%
% Usage Notes
% Input srl can be an array of objects.
%
% See also OPEN_COM_PORT.
%
% Copyright © 2014 Time Domain, Huntsville, AL


for i = 1:length(srl)
  
  get_cfg_rqst(srl(i),i)
  
  Ktry = 0;
  
  while srl(i).BytesAvailable < 48 && Ktry <= 10
    
    Ktry = Ktry + 1;
    pause(0.0001)
    
  end
  
  if Ktry <= 10
    
    msg = uint8(fread(srl(i),srl(i).BytesAvailable,'uint8'));
    
    % Bytes 1:4 are USB synchronization pattern and packet length.
    [cfg,msg_typ,msgID] = parse_msg(msg(5:end));
    
    cfg
    
  else
    fprintf('Configuration data not returned.')
    
  end
  
end

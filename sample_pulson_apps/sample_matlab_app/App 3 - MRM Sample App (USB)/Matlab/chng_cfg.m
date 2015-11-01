function chng_cfg(srl,Tscn,Gtx,PII)
% VIEW_CFG Function to view configuration.
%
% Syntax
% chng_cfg(srl,Tscn,Gtx,PII)
%
% Input
% srl - serial object
% Tscn - two element array with scan start and stop times in ns
% Gtx - transmit gain [0 (low) - 63 (high)]
% PII - pulse integration index (integration = 2^PII)
%
% Output
% NONE
%
% Usage Notes
% Input srl can be an array of objects.
%
% See also OPEN_COM_PORT, VIEW_CFG.
%
% Copyright © 2014 Time Domain, Huntsville, AL


for i = 1:length(srl)
  
  % Read current configuration.
  get_cfg_rqst(srl(i),i)
  
  Ktry = 0;
  
  while srl(i).BytesAvailable < 48 && Ktry <= 10
    
    Ktry = Ktry + 1;
    pause(0.0001)
    
  end
  
  if Ktry <= 10
    
    msg = uint8(fread(srl(i),srl(i).BytesAvailable,'uint8'));
    
    [cfg,msg_typ,msgID] = parse_msg(msg(5:end));
    
  end
  
  % Copy then change the desired configuration parameters.
  CFG = cfg;
  
  CFG.scanStartPs = uint32(Tscn(1)*1000);
  CFG.scanStopPs = uint32(Tscn(2)*1000);
  CFG.transmitGain = uint8(Gtx);
  CFG.pulseIntegrationIndex = uint16(PII);
  CFG.persistFlag = uint8(1);
  
  set_cfg_rqst(srl(i),i,CFG)
  
  Ktry = 0;
  
  while srl(i).BytesAvailable < 12 && Ktry <= 10
    
    Ktry = Ktry + 1;
    pause(0.0001)
    
  end
  
  if Ktry <= 10
    msg = uint8(fread(srl(i),srl(i).BytesAvailable,'uint8'));
  end
  
end

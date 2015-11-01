function [str,msg_typ,msgID] = parse_msg(msg)
% PARSE_MSG Utility function to parse some messages from MRM API.


msg_typ = dec2hex(typecast([msg(2) msg(1)],'uint16'));
msgID = typecast([msg(4) msg(3)],'uint16');

switch msg_typ
  case '1102'
    % MRM_GET_CONFIG_CONFIRM
    str.nodeId = typecast([msg(8) msg(7) msg(6) msg(5)],'uint32');
    str.scanStartPs = typecast([msg(12) msg(11) msg(10) msg(9)],'int32');
    str.scanStopPs = typecast([msg(16) msg(15) msg(14) msg(13)],'int32');
    str.scanStepBins = typecast([msg(18) msg(17)],'uint16');
    str.pulseIntegrationIndex = typecast([msg(20) msg(19)],'uint16');
    str.seg1NumSamples = typecast([msg(22) msg(21)],'uint16');
    str.seg2NumSamples = typecast([msg(24) msg(23)],'uint16');
    str.seg3NumSamples = typecast([msg(26) msg(25)],'uint16');
    str.seg4NumSamples = typecast([msg(28) msg(27)],'uint16');
    str.seg1Multiple = uint8(msg(29));
    str.seg2Multiple = uint8(msg(30));
    str.seg3Multiple = uint8(msg(31));
    str.seg4Multiple = uint8(msg(32));
    str.antennaMode = uint8(msg(33));
    str.transmitGain = uint8(msg(34));
    str.codeChannel = uint8(msg(35));
    str.persistFlag = uint8(msg(36));
    str.timeStamp = typecast([msg(40) msg(39) msg(38) msg(37)],'uint32');
    
  case '1103'
    % MRM_CONTROL_CONFIRM
    str.stat = typecast([msg(8) msg(7) msg(6) msg(5)],'uint32');
    
  case 'F201'
    % MRM_SCAN_INFO
    str.nodeId = typecast([msg(8) msg(7) msg(6) msg(5)],'uint32');
    str.timeStamp = typecast([msg(12) msg(11) msg(10) msg(9)],'uint32');
    str.scanStartPs = typecast([msg(32) msg(31) msg(30) msg(29)],'uint32');
    str.scanStopPs = typecast([msg(36) msg(35) msg(34) msg(33)],'uint32');
    str.scanStepBins =  typecast([msg(38) msg(37)],'uint16');
    str.antennaID = uint8(msg(39));
    str.scanType = uint8(msg(41));
    str.opMode = uint8(msg(42));
    str.messageSamples = typecast([msg(44) msg(43)],'uint16');
    str.totalSamples = typecast([msg(48) msg(47) msg(46) msg(45)],'uint32');
    str.messageIndex = typecast([msg(50) msg(49)],'uint16');
    str.numberMessages = typecast([msg(52) msg(51)],'uint16');
    str.scanData = repmat(int32(0),1,str.messageSamples);
    for i = 1:str.messageSamples
      j = 4*(i - 1);
      k = ((j + 3):-1:j) + 53;
      str.scanData(i) = typecast(msg(k),'int32');
    end
    
  otherwise
    str = [];
    
end

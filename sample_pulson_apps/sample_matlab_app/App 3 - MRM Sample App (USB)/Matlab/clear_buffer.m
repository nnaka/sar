function clear_buffer(srl)
% CLEAR_BUFFER Utility function to clear COM port buffer.


while srl.BytesAvailable > 0
  msg = uint8(fread(srl,srl.BytesAvailable,'uint8'));  
end

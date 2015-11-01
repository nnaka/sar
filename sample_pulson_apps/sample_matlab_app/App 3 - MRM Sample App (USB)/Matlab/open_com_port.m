function srl = open_com_port(num)
% OPEN_COM_PORT Function to open COM ports.
%
% Syntax
% srl = open_com_port(num)
%
% Input
% num - integer COM port number
%
% Output
% srl - serial object
%
% Usage Notes
% Input num can be an array in which output srl is an array of objects.
%
% The COM port number must be determined using tools available on the
% operating system being used. For Windows, the Device Manager shows
% assigned COM devices.
%
% If the output srl gets deleted by accident, the MATLAB function INSTRFIND
% will return objects for all devices.
%
% Copyright © 2014 Time Domain, Huntsville, AL


for i = 1:length(num);
  
  pause(1)
  
  comPort = sprintf('COM%i',num(i));
  
  % The parameters BaudRate, DataBits, Parity, and StopBits are set to
  % valid values for a serial object but have no effect on the serial COM
  % port. The parameter InputBufferSize does effect the serial COM port
  % buffer and must be set large enough to hold the expected number of
  % bytes from incoming messages.
  srl(i) = serial(comPort, ...
    'BaudRate',115200,'DataBits',8,'Parity','none','StopBits',1, ...
    'InputBufferSize',100000, ...
    'BytesAvailableFcnCount', 1, ...
    'BytesAvailableFcnMode', 'byte', ...
    'ReadAsyncMode','continuous' ...
    );
  
  try
    fopen(srl(i));
    fprintf('%s open -> s(%i)\n',comPort,i)
    
  catch err
    disp(err.message)
    
    if i > 1
      fclose(srl(1:i-1));
    end
    delete(srl)
    
    error('Could not open port: %s',comPort)
    
  end
  
end

function [] = processScan(rawCollect, ovsFac, C_mps, rangeScaleFac)

addpath('support');

% Initialized constants and pre-compute variables
numSamp = length(rawCollect{1}.scan); % Number of samples per scan
delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution

% Ranges of each bin (m)
dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); 

% Scale factor applied to the data as a fn of range
rangeScale = dataRange.^(rangeScaleFac); 

% Window applied to reduce range sidelobes
rngWin = hann_window(numSamp); 

for i = 1:length(rawCollect);
    
    tmpRP = rawCollect{i}.scan;
    tmpRP = tmpRP.*rngWin;
    
    % convert to IQ (hilbert transform then modulate to baseband)
    tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:length(tmpRP)-1)).');
    
    % interpolate up, and scale signal vs range
    tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale;
    x = interp(rawCollect{i}.distanceAxis_m, ovsFac)';
    
    % plot each scan incrimentally
    hold on;
    plot(x, abs(tmpRP));
    xlabel('Distance (m)');
    ylabel('Signal Strength');
    grid on;
    
    pause(0.1);
    
end



end 
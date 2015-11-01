warning off;


C_mps = 299792458;
ovsFac = 4;                 	% Oversampling factor applied when interpolating
scanStepBins = 32;          
scanResPs = scanStepBins*1.907; % 61 ps sampling resolution
rangeScaleFac = 3/2;        	% Normalize by r^rangeScaleFac

numSamp = length(rawCollect{1}.scan); % Number of samples per scan
delta_r = scanResPs/1e12*C_mps/2;     % (m) Range resolution

%% Pre-compute Variables

dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); % Ranges of each bin (m)
rangeScale = dataRange.^(rangeScaleFac); % Scale factor applied to the data as a fn of range

rngWin = hann_window(numSamp); % Window applied to reduce range sidelobes

% Process Scan
tmpRP = rawCollect{1}.scan;
tmpRP = tmpRP.*rngWin;
tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).'); % convert to IQ (hilbert transform then modulate to baseband)
tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale; % interpolate up, and scale signal vs range  
x = interp(rawCollect{1}.distanceAxis_m, ovsFac)';

% % Remove signal below ampFilter
% for j=1:(numSamp*ovsFac)
%     if tmpRP(j) < (ampFilter)
%         tmpRP(j) = 0;
%     end
% end


figure(1);
hold all;
plot(x,abs(tmpRP));
xlabel('Distance (m)');
ylabel('Signal Strength');
grid on;
drawnow;
movegui('north')

% % Plot each scan as we go
figure(2);
hold all;
plot(rawCollect{1}.distanceAxis_m,rawCollect{1}.scan);
xlabel('Distance (m)');
ylabel('Signal Strength');
grid on;
drawnow;
movegui('northwest')
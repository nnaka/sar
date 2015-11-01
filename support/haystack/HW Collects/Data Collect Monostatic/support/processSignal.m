%% Initialize Constants

warning off;

% User selectable
ovsFac = 4;                  % Oversampling factor applied when interpolating
rangeScaleFac = 3/2;         % Normalize by r^rangeScaleFac

% Static parameters
C_mps = 299792458;                                  % (m/s) Speed of light
numSamp = length(rawCollect{1}.scan);               % Number of samples per scan
delta_r = rawCollect{1}.scanResPs/1e12*C_mps/2;     % (m) Range resolution

%% Pre-compute Variables

dataRange = (0:ovsFac*numSamp-1).' * (delta_r/ovsFac); % Ranges of each bin (m)
rangeScale = dataRange.^(rangeScaleFac); % Scale factor applied to the data as a fn of range

rngWin = hann_window(numSamp); % Window applied to reduce range sidelobes

%% Process Scans
for i=1:10

	% Remove signal in first 5m
	for j=1:numSamp
		if rawCollect{i}.distanceAxis_m(j) < 5
			rawCollect{i}.scan(j) = 0;
		end
	end

	% Process Scan
	tmpRP = rawCollect{i}.scan;
	tmpRP = tmpRP.*rngWin;
	tmpRP = hilbert(tmpRP).*(exp(-1j*pi/2*(0:(length(tmpRP)-1))).'); % convert to IQ (hilbert transform then modulate to baseband)
	tmpRP = fft_interp(tmpRP, ovsFac) .* rangeScale; % interpolate up, and scale signal vs range  
	x = interp(rawCollect{i}.distanceAxis_m, ovsFac)';

	% Remove signal below 1E7
	for j=1:(numSamp*ovsFac)
		if tmpRP(j) < (1e+7)
			tmpRP(j) = 0;
		end
	end

	% Plot each scan as we go
	figure(5);
	hold all;
	plot(rawCollect{i}.distanceAxis_m,rawCollect{i}.scan);
	xlabel('Distance (m)');
	ylabel('Signal Strength');
	grid on;
	drawnow;

	figure(6);
	hold all;
	plot(x,tmpRP);
	xlabel('Distance (m)');
	ylabel('Signal Strength');
	grid on;
	drawnow;
end
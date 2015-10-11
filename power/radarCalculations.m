% ----------------------------------------------------------------------- %
% Radar Calculations 
% 
% The main piece of code for calculating flight time, velocity, and number
% of pulses required to sweep out an aperture of a given range. 

% It is important to note the tradeoff between maximum range and aperture
% size. As range increases, aperture size must also increase at a
% proportional rate in order to maintain equal range resolution at all
% viewing ranges. Increased aperture size leads to an increased flight
% time, one that quickly exceeds the time air time alotted by the battery.
% Thus it is not always advantageous to image objects at the maximum range
% that is possible.
%
% ----------------------------------------------------------------------- %

clear;

% parameters
c = 3e8;                                % (m/s) speed of light
fc = 4.3e9;                             % ( Hz) center frequency
lambda = c / fc;                        % ( m ) signal wavelength
Ae = .1;                                % (m^2) target cross section area
r = 100;                                % ( m ) range
l = 1;                                  % ( dB) atmospheric loss

pii = 8;
quanta = 8;
scan_time = quanta * .792e-6 * 2^pii;

Pt = 27;                                % (dBm) power transmitted
Gt = 12;                                % ( dB) transmitter gain
Gx = Gt;                                % ( dB) receiver gain
p_gain = get_p_gain(pii);


range = maxRange(Pt, Gt, p_gain) - 20;% ( m ) Maximum detectable range
                                        %       As per above, battery
                                        %       contraints on flight time
                                        %       prohibit imaging objects at
                                        %       maximum range. We lessened
                                        %       the range to reach a flight
                                        %       time within 18.5 minutes

bw = 1.25e9;                            % ( Hz) signal bandwidth
D_r = c / (2 * bw);                     % ( m ) range resolution 
D_yr = D_r;                             % ( m ) cross range resolution, Y
D_xr = D_yr;                            % ( m ) cross range resolution, X 

Dx = 2 * range * lambda * bw / c;       % ( m ) Aperture width
Dy = Dx / 6;                            % ( m ) Aperture height

numPulses = ceil(Dx * 2 / lambda);
numRows = ceil(Dy * 2 / lambda);
totalPulses = numPulses * numRows;

prf = 1/scan_time;

velocity = min([prf * lambda / 2, 8]);  % cap the drone's speed at 8 m/s
row_time = numPulses * lambda / (2 * velocity);

turn_time = 2;                          % (sec) rather arbitrary, should 
                                        %       be tested 

flight_time = (row_time * numRows) + (turn_time * (numRows - 1));


fprintf('Flight time: %f minutes\n', flight_time / 60);
fprintf('UAV Velocity: %f m/s\n', velocity);
fprintf('Total Pulses: %d\n\n\n', totalPulses);



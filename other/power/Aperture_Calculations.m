% ----------------------------------------------------------------------- %
% Aperture Calculations
%
% This script outputs the flight time and velocity for a drone to sweep out
% an aperture with the given antenna gain parameters. 
%
% NOTE: Not the main script for this...radarCalculations is more complete. 
%
% ----------------------------------------------------------------------- %

range = maxRange(41, 12, 20);           % Parameters: transmit gain
lambda = 0.0698;                        %             antenna gain
c = 3e8;                                %             processing gain
bw = 1.25e9;


Dx = 2 * range * lambda * bw / c;       % Aperture row dimensions
Dy = Dx / 6;                            % Aperture col dimensions

numPulses = ceil(Dx * 2 / lambda);
numRows = ceil(Dy * 2 / lambda);

prf = 200;                              % set somewhat arbitrarily
                                        % Need a better metric 
row_time = numPulses / prf;

turn_time = 2;

flight_time = (row_time * numRows) + (turn_time * (numRows - 1));
velocity = prf * lambda / 2;

fprintf('Flight time: %f minutes\n', flight_time / 60);
fprintf('UAV Velocity: %f m/s\n', velocity);

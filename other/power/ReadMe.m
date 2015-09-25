% ----------------------------------------------------------------------- %
% Power Read Me 
%
% This folder contains functions and scripts that juggle radar and drone
% power parameters to calculate aperture sizes, flight times, and maximum
% radar ranges.
%
% Functions: 
%
%   Aperture_Calculations.m - Given a set of radar parameters, outputs the
%                             required flight time and drone velocity to
%                             sweep out the needed aperture.
%                             radarCalculations.m is more complete.
%
%   get_p_gain.m            - Returns the processing gain for the amplifier
%                             based on the number of pulses taken per
%                             location. Processing Gain is amount of singal
%                             gain achieved by integrating multiple pulses
%                             taken at the same location in an aperture.
%
%   maxRange.m              - The first version of a function that returns
%                             the maximum visible radar range given the
%                             amplifier characteristics of transmit power,
%                             gain, and processing gain.
%
%   maxRangeV2.m            - The better version of maxRange. Trust me. 
%
%   radarCalculations.m     - The main piece of code for calculating drone
%                             flight time, velocity, and number of pulses
%                             required to fully sweep out an aperture.
%
% ----------------------------------------------------------------------- %
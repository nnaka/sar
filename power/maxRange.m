function [maxRng] = maxRange(Pt, Gt, p_gain)

% ----------------------------------------------------------------------- %
% maxRange
% 
% Given the parameters of signal gain, antenna gain, and processing gain,
% this function will output the maximum possible range that the PulsOn p410
% can detect 10 cm^2 object 
% 
% Pt,     Signal Gain     - (dBm) gain from the PulsOn unit itself plus an 
%                                 external amplifier.
% Gt,     Antenna Gain    - (dB ) gain inherent to the antenna plus gain 
%                                 from an external amplifier.
% 
% NOTE: maxRangeV2 is the most up to date version of this code.
%
% ----------------------------------------------------------------------- %

if nargin == 0
    Pt = 41;
    Gt = 12;
    p_gain = 20;
end

Gx = Gt;                            % ( dB) receiver gain
c = 3e8;                            % (m/s) speed of light
fc = 4.3e9;                         % ( Hz) center frequency
bw = 1.25e9;                        % ( Hz) signal bandwidth
fqs = linspace((fc - bw/2),(fc + bw/2),10000);
lambda = c / fc;                    % ( m ) signal wavelength
Ae = .1;                            % (m^2) target cross section area
r = 100;                            % ( m ) range
l = 1;                              % ( dB) atmospheric loss

% convert gain in dB to gain in watts 
Gt_w = 10.^(Gt./10);
Gx_w = Gt_w;
Pt_w = 10^(Pt/10) / 1000;

% atmospheric noise 
k = 1.3806488 * 10^-23;             % Boltzmann constant
t = 273.15 + 25;                    % degrees kelvin
thermal_noise = k * t * bw;
thermal_dB = 10*log10(thermal_noise);

% Radar equation: Power received 
num = Pt_w .* Gt_w .* Gx_w * p_gain * lambda^2 * Ae;
den = (4 * pi)^3 * r^4 * l;
Pr = num ./ den;
Pr_dB = 10*log10(Pr);
Pr_dB_min = thermal_dB + 10;        % to be detected, the signal must be 
Pr_min = 10.^(Pr_dB_min / 10);      % 10dB above thermal noise

% Radar equation: maximum range 
den = (4 * pi)^3 * Pr_min * l;
maxRng = nthroot(num ./ den, 4);

% fprintf('Max Range: %f\n', maxRng);

end 


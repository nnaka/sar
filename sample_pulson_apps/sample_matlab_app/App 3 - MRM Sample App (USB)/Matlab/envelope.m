function y = envelope(x)
% ENVELOPE Utility function to compute envelope of scan.


% Butterworth filter.
% [b,a] = butter(6,0.4);  NOTE: Signal Processing Toolbox function

a = [
  1.0000000000
 -1.1876006802
  1.3052133493
 -0.6743275253
  0.2634693483
 -0.0517530339
  0.0050225266
];

b = [
  0.0103128748
  0.0618772486
  0.1546931214
  0.2062574953
  0.1546931214
  0.0618772486
  0.0103128748
];

Nord = length(a) - 1;
Nshft = Nord/2 + 1;

y = sqrt(2*max(filter(b,a,x.^2),0));
y = [y(Nshft:end) zeros(1,Nshft-1)];

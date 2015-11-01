function y = movingAvg(x)
% Finite Impulse Response Low-pass Filter Order 5
% good for creating an envelope
%
% NOTES
% Stupid box car averager because I don't have time to design better.
% But does a fine job of smushing out the bumps.

b(1) = 1/6;
b(2) = 1/6;
b(3) = 1/6;
b(4) = 1/6;
b(5) = 1/6;
b(6) = 1/6;

N = length(x);
y = zeros(1,N);

for k = 6:N
  y(k) = ...
    + b(1)*x(k) ...
    + b(2)*x(k-1) ...
    + b(3)*x(k-2) ...
    + b(4)*x(k-3) ...
    + b(5)*x(k-4) ...
    + b(6)*x(k-5);
end

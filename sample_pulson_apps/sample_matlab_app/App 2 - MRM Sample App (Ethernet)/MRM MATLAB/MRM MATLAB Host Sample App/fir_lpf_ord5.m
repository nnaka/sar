function y = fir_lpf_ord5(x)
% Finite Impulse Response Low-pass Filter Order 5
% butter(x,a,b) coefficients
% Made explicit for future java coding

b(1) = 0.6;
b(2) = 0.3;
b(3) = 0.6;
b(4) = 0.6;
b(5) = 0.3;
b(6) = 0.6;

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
%y = y/sum(b); % normalize

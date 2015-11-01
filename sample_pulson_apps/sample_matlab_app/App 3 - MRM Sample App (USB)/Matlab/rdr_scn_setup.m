function [T1,T2,R1,R2,Rbin,Nbin] = rdr_scn_setup(R1,R2)
% RDR_SCN_SETUP Function to setup scan times and associated range bins.
%
% Syntax
% [T1,T2,R1,R2,Rbin,Nbin] = rdr_scn_setup(R1,R2)
%
% Input
% R1 - start range (m)
% R2 - stop range (m)
%
% Output
% T1 - start time (ns)
% T2 - stop time (ns)
% R1 - start range (m) [adjusted]
% R2 - stop range (m) [adjusted]
% Rbin - scan range array (m)
% Nbin - number of scan bins
%
% Usage Notes
% R1 will be set as close as possible to the desired value based on timing
% precision in the radio. R2 will be rounded to the next higher possible
% value with the constraint that scans must be a multiple of 96 bins, which
% is equivalent to approximately 0.88 m.
%
% Copyright © 2014 Time Domain, Huntsville, AL


% Speed of light.

c = 0.29979;  % m/ns

% Radio parameters.

dTmin = 1/(512*1.024);  % ns

Tbin = 32*dTmin;  % ns

dNbin = 96;  % number of bins in a scan segment

dT0 = 10;  % ns
% NOTE: This is the empirically measured value with broadspec antennas
% connected directly on SMA connectors as provide in the kit. Alternate
% antenna connections and/or cable lengths require a different value.

% Calculation of T1 and T2 subject to radio timing.
% NOTE: This code attempts to restrict times similar to MRM to compute
% required scan start and stop times but the actual radio values may be
% slightly different.

T1 = 2*R1/c + dT0;  % ns
T2 = 2*R2/c + dT0;  % ns

Nbin = (T2 - T1)/Tbin;
Nseg = ceil(Nbin/dNbin);
Nbin = dNbin*Nseg;

T1 = floor(1000*dTmin*floor(T1/dTmin))/1000;  % ns

T2 = Nbin*Tbin + T1;  % ns
T2 = floor(1000*dTmin*ceil(T2/dTmin))/1000;  % ns

% Recompute R1 and R2 using T1 and T2.

R1 = c*(T1 - dT0)/2;
R2 = c*(T2 - dT0)/2;

dRbin = c*Tbin/2;  % m 

Rbin = R1 + dRbin*(0:Nbin-1);  % m

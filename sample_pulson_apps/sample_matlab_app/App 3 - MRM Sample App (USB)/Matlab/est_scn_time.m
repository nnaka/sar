function Tscn = est_scn_time(Nbin,PII)
% EST_SCN_TIME Function to estimate time to collect single scan.
%
% Syntax
% Tscn = est_scn_time(Nbin,PII)
%
% Input
% Nbin - number of scan bins
% PII - pulse integration index (integration = 2^PII)
%
% Output
% Tscn - scan time (ms)
%
% Usage Notes
% Estimate of scan time using the nominal pulse period of 100 ns. The
% actual scan time depends on the channel selected.
%
% See also RDR_SCN_SETUP.
%
% Copyright © 2014 Time Domain, Huntsville, AL


dNbin = 96;  % number of bins in a scan segment

Nseg = Nbin/dNbin;

Nstp = 8;  % radio parameter

Tpls = 100;  % ns (nominal pulse period; depends on channel)

Tscn = Nseg*(Nstp*(2^PII)*Tpls)*1e-6;  % ms

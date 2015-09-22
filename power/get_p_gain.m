function [gain] = get_p_gain(pii)

% ----------------------------------------------------------------------- %
% get_p_gain
%
% Processing gain is the gain you get from integrating many radar pulses
% taken at a specific location. With more pulses taken at a given location,
% the SNR is lowered at the cost of requiring a slower flight time.
% 
% Processing gain is a linear function based on the pulse integration
% index (pii). A large pii yields a larger processing gain.
%
% This function is used by:
%       maxRangeV2
%       radarCalculations
% ----------------------------------------------------------------------- %

gain = ceil(pii)*3 + 30;

end 

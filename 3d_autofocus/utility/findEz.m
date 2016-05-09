% Returns the total image energy of the complex image Z given the magnitude of
% the pixels in Z
function [ Ez ] = findEz( Z_mag )
  Ez = sum(Z_mag);
end

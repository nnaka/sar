% Returns the entropy of the complex image `Z`
function [ entropy ] = H( Z )
  Z_mag = Z .* conj(Z);         
  Ez = findEz(Z_mag);

  Z_intensity = Z_mag / Ez;
  % TODO: (joshpfosi) Why is this negated?
  entropy = - sum(Z_intensity .* log(Z_intensity));
end

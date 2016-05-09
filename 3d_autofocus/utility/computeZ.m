% Defines z_vec(phi), where B is a 1D representation of the image as described
% above.
function [ Z ] = computeZ(phi_offsets, B)
  K = numel(phi_offsets);
  N = length(B) / K;
  
  % Form 1D array of e^-j * phi_i which repeats every kth element to allow for
  % simple elementwise multiplication on B. See equation (2) in 'tech_report.pdf'.
  arr = repmat(exp(-1j * phi_offsets), 1, N);
  
  % `reshape(B .* arr, K, [])` returns a matrix with `N` columns and `K` rows.
  % Each column vector contains each of `K` contributions to the pixel `i`, so
  % summing each column vector results in `Z`.
  Z = sum(reshape(B .* arr, K, []), 1);
end

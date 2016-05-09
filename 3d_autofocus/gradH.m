% TODO: Nice doc comments
function [ grad ] = gradH( phi_offsets, B, delta )
  addpath('utility');

  K = numel(phi_offsets);
  grad = zeros(1, K);

  delta = 0.001; % Arbitrary constant for finite difference

  % k x k identity matrix in MATLAB
  ident = eye(K);

  fprintf('In gradH, about to compute Z\n');
  Z = computeZ(phi_offsets, B);
  fprintf('Computed Z\n');
  H_not = H(Z);
  fprintf('Computed H_not\n');

  parfor k = 1:K
    Z = computeZ(phi_offsets + transpose(ident(:, k) * delta), B);
    grad(k) = (H(Z) - H_not) / delta;
  end
end

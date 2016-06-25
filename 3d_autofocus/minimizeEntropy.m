% -----------------------------------------------------------------------------
% The following is a MATLAB implementation of the standard gradient descent
% minimization of the image entropy cost function. The below algorithm
% implements the technique described analytically in 'tech_report.pdf'.
%
% @param B [K * N array] pulse history formatted as a 1D array of length `K * N`
% array where `K` refers to the number of pulses in the aperture and `N` refers
% to the number of pixels in either the 2D or 3D image. The array should be
% formatted such that the `i`th set of `K` elements correspond to the
% contributions to pixel `i`.  E.g. the first `K` elements represent each
% pulse's contribution to pixel 0.
%
% @param K [Integer] number of pulses to form `B`
%
% @return focusedImage [Array] X by Y (by Z) focused image
% @return minEntropy [Float] entropy of focused `B`
% @return origEntropy [Float] entropy of unfocused image
% -----------------------------------------------------------------------------
% TODO: `delta` should be a parameter to gradFunc
function [ focusedImage, minEntropy, origEntropy ] = minimizeEntropy( B, K )
  s                 = 100;  % Step size parameter for gradient descent
  convergenceThresh = 0.001; % Difference after which iteration "converges"
  stepMinimum       = 0.01; % Minimum step size

  l = 2; % First iteration is all 0s, so start at iteration 2

  % Holds array of potentially minimizing phase offsets (guessing zero
  % initially). 100 is an arbitrary guess for the number of iterations
  phiOffsets = zeros(100, K);

  focusedImage = computeZ(phiOffsets(1, :), B);
  minEntropy   = H(focusedImage);
  origEntropy  = minEntropy;

  while (1) % phiOffsets(1) = 0
    grad1 = gradH(phiOffsets(l - 1, :), B);
    grad2 = gradHAnalytical(phiOffsets(l - 1, :), B);

    phiOffsets(l, :) = phiOffsets(l - 1, :) - s * grad2;

    tempImage = computeZ(phiOffsets(l, :), B);
    tempEntropy = H(tempImage);
    
    fprintf('tempEntropy = %d, minEntropy = %d\n', tempEntropy, minEntropy);

    if (minEntropy < tempEntropy)
        s = s / 2;

        fprintf('Reducing step size to %d\n', s);

        if (s < stepMinimum)
          fprintf('s is below minimum so breaking\n');
          break;
        end
    else
        if (minEntropy - tempEntropy < convergenceThresh) 
          fprintf('Change in entropy (%d - %d = %d) < %d\n', ...
                   minEntropy, tempEntropy, minEntropy - tempEntropy, ...
                   convergenceThresh);
          break; % if decreases in entropy are small
        end

        minEntropy   = tempEntropy;
        focusedImage = tempImage;

        l = l + 1;
    end
  end
end

function [grad] = gradHAnalytical( phi_offsets, B )
  K = numel(phi_offsets);
  N = length(B) / K;

  Z = computeZ(phi_offsets, B);
  Z_mag = Z .* conj(Z);         

  Ez = findEz(Z_mag);
  fprintf('Ez=%d\n', Ez);
  
  grad = zeros(1, K);
  
  sum = 0;
  for i = 1:N
    sum = sum + Z_mag(i) * log(Z_mag(i));
  end

  for k = 1:K
     acc = 0;

     for n = 1:N
        partial = partialZ(Z(n), phi_offsets, B, k, n);
        
        term = 1 / (Ez * Ez) * sum - (1 / Ez) * log(Z_mag(n));
        acc = acc + term * partial;
      end

      grad(k) = acc / Ez;
  end
end

function [ p ] = partialZ(Zn, phi_offsets, B, k, n)
  K = numel(phi_offsets);
  
  pz = 1i * B((n-1) * K + k) * exp(1i * phi_offsets(k));

  p = pz * conj(Zn) + conj(pz) * Zn;
end

% TODO: Nice doc comments
function [ grad ] = gradH( phi_offsets, B )
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

% Returns the entropy of the complex image `Z`
function [ entropy ] = H( Z )
  Z_mag = Z .* conj(Z);         
  Ez = findEz(Z_mag);

  Z_intensity = Z_mag / Ez;
  % TODO: (joshpfosi) Why is this negated?
  entropy = - sum(Z_intensity .* log(Z_intensity));
end

% Returns the total image energy of the complex image Z given the magnitude of
% the pixels in Z
function [ Ez ] = findEz( Z_mag )
  Ez = sum(Z_mag);
end

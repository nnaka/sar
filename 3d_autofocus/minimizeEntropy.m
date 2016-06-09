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
  addpath('utility');

  s                 = 100;  % Step size parameter for gradient descent
  convergenceThresh = 0.001; % Difference after which iteration "converges"
  stepMinimum       = 0.01; % Minimum step size

  l = 2; % First iteration is all 0s, so start at iteration 2

  % Holds array of potentially minimizing phase offsets (guessing zero
  % initially). 50 is an arbitrary guess for the number of iterations
  phiOffsets = zeros(50, K);

  focusedImage = computeZ(phiOffsets(1, :), B);
  minEntropy   = H(focusedImage);
  origEntropy  = minEntropy;

  while (1) % phiOffsets(1) = 0
    phiOffsets(l, :) = phiOffsets(l - 1, :) - s * gradH(phiOffsets(l - 1, :), B);

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


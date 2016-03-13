% The following is a MATLAB implementation of the standard gradient descent
% minimization of the image entropy cost function. The below algorithm
% implements the technique described analytically in 'tech_report.pdf'.
%
% B is a 4D array of b_k values
% L is the number of iterations
function [ out, minEntropy, maxEntropy ] = minEntropyGradientC( B, L )
  THRESHOLD = 0.05;
  MAX_ITER = 50;
  X = size(B,1); Y = size(B,2); Z = size(B,3); K = size(B,4);
  l = 2;
  minIdx = 1;
  minEntropy = Inf;

  % Holds array of potentially minimizing phase offsets - 100 is an arbitrary
  % maximum number of iterations
  %
  % Guess zero initially
  phi_offsets = zeros(MAX_ITER, K);

  % Step size parameter for gradient descent
  s = 100;
  
  % As iterating over a 4D array reduces spatial locality, convert `B` once
  % into a 1D array and then convert back after minimization of phi is
  % complete. 
  % TODO: (joshpfosi) Could potentially use `cat` here
  B_tmp = [];
  for x = 1:X
      for y = 1:Y
          for z = 1:Z
                B_tmp = horzcat(B_tmp, reshape(B(x,y,z,:), 1, K));
          end
      end
  end
  
  B = B_tmp;
  clear('B_tmp');

  maxEntropy = H(image(phi_offsets(1, :), B));

  while (1) % phi_offsets(1) = 0
    phi_offsets(l, :) = phi_offsets(l - 1, :) - s * grad_h_mex(phi_offsets(l - 1, :), B);
    focusedImage = image(phi_offsets(l, :), B);
    tempEntropy = H(focusedImage);
    
    fprintf('tempEntropy = %d, minEntropy = %d\n', tempEntropy, minEntropy);

    if (minEntropy < tempEntropy)
        s = s / 2;

        fprintf('Reducing step size to %d\n', s);

        if (s < THRESHOLD)
          fprintf('s is below threshold so breaking');
          break;
        end
    else
        if (minEntropy - tempEntropy < THRESHOLD) 
          fprintf('%d - %d = %d < 0.001\n', minEntropy, tempEntropy, minEntropy - tempEntropy);
          break; % if decreases in entropy are small
        end

        minIdx = l;
        minEntropy = tempEntropy;
        l = l + 1;
    end
  end
  
  % `focusedImage` now contains the 1D representation of the entropy-minimized
  % B, constructed using phase offsets `phi_offets(minIdx)`. We must reshape it
  % back into a 3D array.
  % TODO: (joshpfosi) Use `reshape` instead of ugly `for`s.
  out = zeros(X,Y,Z);
  for x = 1:X
      for y = 1:Y
          for z = 1:Z
              % We must index into the 1D array as if it were 3D so the index
              % is complicated.
              idx = (x - 1) * Y * Z + (y - 1) * Z + (z - 1) + 1;
              out(x,y,z) = focusedImage(idx);
          end
      end
  end
end

% Defines z_vec(phi), where B is a 1D representation of the image as described
% above.
function [ Z ] = image(phi_offsets, B)
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

  Ez = sum(Z_mag);

  Z_intensity = Z_mag / Ez;
  % TODO: (joshpfosi) Why is this negated?
  entropy = - sum(Z_intensity .* log(Z_intensity));
end

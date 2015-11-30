% B is a 4D array of b_k values
% L is the number of iterations
function [ focusedImage, minEntropy ] = minEntropyFminunc( B, L )
  % Guess zero initially
  l = 2;
  minIdx = 1;
  minEntropy = 100;
  phi_offsets = zeros(100, size(B, 4));

  % Step size parameter for gradient descent
  s = 10;

  while (1) % phi_offsets(1) = 0
    phi_offsets(l, :) = phi_offsets(l - 1, :) - s * gradH(phi_offsets(l - 1, :), B);
    
    
    focusedImage = image(phi_offsets(l, :), B);
    tempEntropy = H(focusedImage);
    
    fprintf('tempEntropy = %d, minEntropy = %d\n', tempEntropy, minEntropy);
    if (tempEntropy < minEntropy)
        minIdx = l;
        minEntropy = tempEntropy;
    else
        break;
    end
    
    l = l + 1;
  end
end

function [grad] = gradH(phi_offsets, B)
    K = numel(phi_offsets);
    grad = zeros(1, K);

    delta = 1; % arbitrary constant for finite difference

    % k x k identity matrix in MATLAB
    ident = eye(K);

    fprintf('In gradH, about to compute Z\n');
    Z = image(phi_offsets, B);
    fprintf('Computed Z\n');
    H_not = H(Z);
    fprintf('Computed H_not\n');

    for k = 1:K
      fprintf('Computing Z for k=%d\n', k);
      Z = image(transpose(phi_offsets) + ident(:, k) * delta, B);
      grad(k) = (H(Z) - H_not) / delta;
    end
end

function [entropy] = H(Z)
  Z_mag = Z .* conj(Z);         
  Ez = findEz(Z);                   

  Z_intensity = Z_mag / Ez;
  entropy = - sum(sum(sum(Z_intensity .* log(Z_intensity))));
end

function [out] = image(phi_offsets, B) % defines z_vec(phi)
  X = size(B, 1); Y = size(B, 2); Z = size(B, 3);
  out = zeros(X, Y, Z);
  K = size(B, 4);

  for x = 1:X
      for y = 1:Y
          for z = 1:Z
              for k = 1:K
                out(x,y,z) = out(x,y,z) + sum(B(x,y,z,k) * exp(-1j * phi_offsets(k)));
              end
          end
      end
  end
end

function [Ez] = findEz(Z)
  Ez = sum(sum(sum(Z .* conj(Z))));
end

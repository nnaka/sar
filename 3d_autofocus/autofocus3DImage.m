% -----------------------------------------------------------------------------
% `autofocus3DImage` is a wrapper for gradient descent entropy minimization.
%
% @param ph [Array] A 3D pulse history of dimensions X by Y by Z by K.
% @param minimizer [function_handle] Function handle use to execute gradient
% descent. This parameter can be one of `minimize_entropy_mex` or
% `minimizeEntropy`. Note that the former can be linked against a C++ or a CUDA
% object file.
%
% @return image [Array] X by Y by Z focused image
% @return minEntropy [Float] entropy of focused image
% @return origEntropy [Float] entropy of unfocused image
% -----------------------------------------------------------------------------
% TODO: camelCase everything, this is MATLAB, afterall
% TODO: Validate arguments
function [ image, minEntropy, origEntropy ] = autofocus3DImage( ph, minimizer )
  ph = single(ph); % Use floats to save memory

  X = size(ph, 1);
  Y = size(ph, 2);
  Z = size(ph, 3);
  K = size(ph, 4);

  % As iterating over a 4D array reduces spatial locality, convert `B` once
  % into a 1D array and then convert back after minimization of phi is
  % complete. 
  % TODO: (joshpfosi) Could potentially use `cat` here
  B = [];
  for x = 1:X
    for y = 1:Y
      for z = 1:Z
        B = horzcat(B, reshape(ph(x,y,z,:), 1, K));
      end
    end
  end

  [focusedImage, minEntropy, origEntropy] = minimizer(B, K);

  % `focusedImage` now contains the 1D representation of the entropy-minimized
  % B, constructed using phase offsets `phi_offets(minIdx)`. We must reshape it
  % back into a 3D array.
  % TODO: (joshpfosi) Use `reshape` instead of ugly `for`s.
  image = zeros(X, Y, Z);
  for x = 1:X
    for y = 1:Y
      for z = 1:Z
        % We must index into the 1D array as if it were 3D so the index
        % is complicated.
        idx = (x - 1) * Y * Z + (y - 1) * Z + (z - 1) + 1;
        image(x, y, z) = focusedImage(idx);
      end
    end
  end
end

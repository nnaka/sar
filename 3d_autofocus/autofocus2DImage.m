% -----------------------------------------------------------------------------
% This function performs identically to its 3D counterpart except returns a 2D
% image.
%
% See `autofocus3DImage.m` for parameter documentation.
% -----------------------------------------------------------------------------
function [ image, minEntropy, origEntropy ] = autofocus2DImage( ph, minimizer )
  X = size(ph, 1);
  Y = size(ph, 2);
  K = size(ph, 3);

  % As iterating over a 3D array reduces spatial locality, convert `B` once
  % into a 1D array and then convert back after minimization of phi is
  % complete. 
  % TODO: (joshpfosi) Could potentially use `cat` here
  B = [];
  for x = 1:X
    for y = 1:Y
      B = horzcat(B, reshape(ph(x,y,:), 1, K));
    end
  end

  [focusedImage, minEntropy, origEntropy] = minimizer(B, K);

  % `focusedImage` now contains the 1D representation of the entropy-minimized
  % B, constructed using phase offsets `phi_offets(minIdx)`. We must reshape it
  % back into a 2D array.
  % TODO: (joshpfosi) Use `reshape` instead of ugly `for`s.
  image = zeros(X, Y);
  for x = 1:X
    for y = 1:Y
      % We must index into the 1D array as if it were 3D so the index
      % is complicated.
      idx = (x - 1) * Y + (y - 1) + 1;
      image(x, y) = focusedImage(idx);
    end
  end
end


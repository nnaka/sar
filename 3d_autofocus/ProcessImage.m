% ----------------------------------------------------------------------- %
% Process Image
% 
% Autofocus the image if the pulse set has the proper number of dimensions
%
% ----------------------------------------------------------------------- %

function [] = ProcessImage( filename )

load(filename);

unfocusedImage = sum(imageSet, 4);
save(strcat(filename, '_image_UNFOCUSED.mat'), 'unfocusedImage', '-v7.3')

tic
[focusedImageC, minEntropyC, origEntropyC] = autofocus3DImage(imageSet, @grad_h_mex);
et2 = toc;
save(strcat(filename, '_image_GRADIENT_C.mat'), 'focusedImageC', '-v7.3')

tic
[focusedImageMatlab, minEntropyMatlab, origEntropyMatlab] = autofocus3DImage(imageSet, @gradH);
et3 = toc;
save(strcat(filename, '_image_GRADIENT_MATLAB.mat'), 'focusedImageMatlab', '-v7.3')

fprintf('GRADIENT C:      Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyC, origEntropyC, et2);
fprintf('GRADIENT MATLAB: Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyMatlab, origEntropyMatlab, et3);

end

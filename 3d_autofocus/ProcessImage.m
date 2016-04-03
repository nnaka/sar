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

% tic
% [focusedImageBrute, minEntropyBrute, maxEntropyBrute] = minEntropyAutoFocus(imageSet, 4);
% et1 = toc;
% save(strcat(filename, '_image_BRUTE.mat'), 'focusedImageBrute', '-v7.3')

tic
[focusedImageC, minEntropyC, maxEntropyC] = minEntropyGradientC(imageSet);
et2 = toc;
save(strcat(filename, '_image_GRADIENT_C.mat'), 'focusedImageC', '-v7.3')

tic
[focusedImageMatlab, minEntropyMatlab, maxEntropyMatlab] = minEntropyGradientMatlab(imageSet);
et3 = toc;
save(strcat(filename, '_image_GRADIENT_MATLAB.mat'), 'focusedImageMatlab', '-v7.3')

% fprintf('BRUTE:           Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyBrute, maxEntropyBrute, et1);
fprintf('GRADIENT C:      Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyC, maxEntropyC, et2);
fprintf('GRADIENT MATLAB: Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyMatlab, maxEntropyMatlab, et3);

end

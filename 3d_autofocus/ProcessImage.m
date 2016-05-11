% -----------------------------------------------------------------------------
% Autofocus image and time results
%
% @param path [String] path to .mat file which loads a variable `pulseHistory`
% into the workspace
% -----------------------------------------------------------------------------

function processImage( path )

load(path);

unfocusedImage = sum(pulseHistory, 4);
save(strcat(path, '_image_UNFOCUSED.mat'), 'unfocusedImage', '-v7.3')

tic
[focusedImageC, minEntropyC, origEntropyC] = autofocus3DImage(pulseHistory, @grad_h_mex);
et2 = toc;
save(strcat(path, '_image_GRADIENT_C.mat'), 'focusedImageC', '-v7.3')

tic
[focusedImageMatlab, minEntropyMatlab, origEntropyMatlab] = autofocus3DImage(pulseHistory, @gradH);
et3 = toc;
save(strcat(path, '_image_GRADIENT_MATLAB.mat'), 'focusedImageMatlab', '-v7.3')

fprintf('GRADIENT C:      Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyC, origEntropyC, et2);
fprintf('GRADIENT MATLAB: Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyMatlab, origEntropyMatlab, et3);

end

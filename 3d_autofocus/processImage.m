% -----------------------------------------------------------------------------
% Autofocus image and time results
%
% @param path [String] path to .mat file which loads a variable `pulseHistory`
% into the workspace
% -----------------------------------------------------------------------------

function processImage( path )

load(path);

tic
[focusedImageMatlab, minEntropyMatlab, origEntropyMatlab] = autofocus2DImage(pulseHistory, @minimizeEntropy);
et3 = toc;

fprintf('GRADIENT: Min Entropy: %f, Orig Entropy: %f, Elapsed Time: %f\n', minEntropyMatlab, origEntropyMatlab, et3);

end

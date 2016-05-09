% Translates GOTCHA Volumetric dataset into format usable by the provided
% backprojection algorithm, here
% (https://dlmacph-radar.googlecode.com/hg/SPIE10toolbox.pdf)
%
% @param path [String] - path to GOTCHA data (terminated by `path*`)
% @param start [Integer] - start angle (1 to 360)
% @param delTheta [Integer] - number of degrees to sweep through (1 to 360)
% @return image [2D array] - complex valued SAR product
%
% TODO: Validate parameters (e.g. start=355, delTheta=20 is an unchecked
% error currently)
function [image, pulseHistory] = bpDriver(path, start, delTheta)

bpData = {};
image = [];
pulseHistory = [];

for i = start:(start + delTheta - 1)
    filename = sprintf('%s/HH/data_3dsar_pass8_az%03i_HH.mat', path, i);

    load(filename);

    % TODO: Scene and image size should be computed
    [X, Y] = meshgrid(-50:0.2:50);

    bpData.minF   = data.freq(1) * ones(1, size(data.fp, 2));
    bpData.x_mat  = X;
    bpData.y_mat  = Y;
    bpData.z_mat  = zeros(size(X, 1));
    bpData.AntX   = data.x;
    bpData.AntY   = data.y;
    bpData.AntZ   = data.z;
    bpData.R0     = data.r0;
    bpData.phdata   = data.fp;

    % Rule of thumb from paper -- power of 2 is optimized
    bpData.Nfft   = 2 ^ nextpow2(10 * size(data.freq, 1));
    bpData.deltaF = (data.freq(end) - data.freq(1)) / (length(data.freq) - 1);

    [bpData, ph] = bp(bpData);

    if size(image, 1) == 0
      image = bpData.im_final;
      pulseHistory = ph;
    else
      image = image + bpData.im_final;
      pulseHistory = horzcat(pulseHistory, ph);
    end
end

% TODO: This should be removed
figure;
hAx = axes;
imagesc(mag2db(abs(image)), 'Parent', hAx);
colormap(hAx,gray);
colorbar;
xlabel('X (meters)');
ylabel('Y (meters)');
set(hAx,'YDir','normal');

end

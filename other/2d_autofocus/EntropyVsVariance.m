% EntropyVsVariance
% by Colin Watts and Ian Fletcher
%
%   This function generates simulated radar returns with unknown error
%   introduced, forms an image and then focuses the image. The amount of
%   error introduced is a function of sigma. To
%   generate a set of data for many different errors set sigma to be a
%   vector. 
%
%   At sigma = 1 the error added is typical of what was found in GPS
%   testing.


%   Quite honestly I mostly use this code to generate images all at once
%   and then focus the images seperately. 
function [  ] = EntropyVsVariance(  )
    printFromFile = false;
    sigma = 1;
    E = zeros(1,length(sigma));
    if ~printFromFile
        for i = 1:length(sigma)
            fprintf('%d: Variance: %d, Test: %d / %d\n', 1, sigma(i), i, length(sigma));
            PulsOnTest(sigma(i));
            RC = format_yOut();
            monoSARwFocus(RC);
            E(i) = E(i)+autoFocus();
        end
    else
        load('MeanEntropy.mat');
    end
    
    plot(sigma,E);
    xlabel('Variance In X, Y and Z positions (mm)')
    ylabel('Entropy')
    %save('MeanEntropy.mat', 'E');
end


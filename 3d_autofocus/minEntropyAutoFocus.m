function [ focused_image, minEntropy ] = minEntropyAutoFocus( pulseSet, numIterations )

% Minimum Entropy Auto Focus
%
% ----------------------------------------------------------------------- %
% Entropy is a measure of image intensity. When a radar image is formed
% from out of phase pulse returns, the image is corrupted and entropy
% increases.The goal of minimum entropy auto focus is to find the phase 
% correction for each pulse return that minimizes the overall entropy of 
% the image. 
% 
% The algorithm works as follows: 
%       1 - One pulse at a time, find the phase offset that minimizes the
%           overall entropy of the image. This is currently done using
%           brute force. 
%       2 - Recalculate the image and entropy using the new pulse phase
%           offsets.
%       3 - Iterate. 
%
% The function argument pulseSet is a 4D Matrix of radar pulse returns.
%
%  Dimension:
%     1:3 - The 3D image that corresponds to each given pulse individaully
%       4 - The pulses themselves
%
%  To create the full radar image, sum pulseSet along the 4th dimension.
%
% The second argument is the number of iterations to perform. Each
% iteration will focus the image slightly more, however there is a rate of
% diminishing returns. 
%
% TODO: 
%       1 - Coordinate descent will converge far faster than any brute 
%           force method. This needs to be implemented. We had hang ups 
%           with determining the correct complex derivative for the 
%           coordinate descent function. Consulting the math department 
%           should help. 
%       2 - The phase offset for each pulse in a given iteration is 
%           calculated independently of all the others. This algorithm can 
%           easily be parallelized.
% ----------------------------------------------------------------------- %

scanSet = pulseSet;

rawImg = sum(scanSet,4);

originalEntropy = findEntropy(rawImg);

% estimated phase correction for each pusle in aperture - begin at 0

numPulses = size(scanSet,4);

scan_iter = scanSet;
phi_offsets = zeros(1,size(scanSet, 4));    % Current phase offsets for each pulse    
phiset = linspace(-1*pi,1*pi, 100);         % All possible phase offsets


% perform the coordinate descent iterations
allEntropies = zeros(1, numPulses*numIterations);
for iter = 1:numIterations
    
    for pulseIdx = 1:numPulses
        
        fprintf('Pulse %d of %d, iteration %d of %d\n', pulseIdx, numPulses, iter, numIterations);

        % Use brute force method to determine the phase offset that
        % minimizes image entropy
        minIdx = 0;
        minEntropy = Inf;
        for i = 1:numel(phiset)
            
            % inject a new phase offset
            scan_iter(:,:,:,pulseIdx) = scanSet(:,:,:,pulseIdx) * exp(1j*phiset(i));
            newImg = sum(scan_iter,4);
            currentEntropy = findEntropy(newImg);
            
            % determine the phase offset that yields the miminum image
            % entropy
            if currentEntropy < minEntropy
                minEntropy = currentEntropy;
                minIdx = i;
            end 
            phi_offsets(pulseIdx) = phiset(minIdx);         
        end

        
        scan_iter(:,:,:,pulseIdx) = scanSet(:,:,:,pulseIdx) * exp(1j*phi_offsets(pulseIdx));
        allEntropies(pulseIdx + (iter-1)*numIterations) = findEntropy(sum(scan_iter,4));
        
        % fprintf('Entropy: %f\n', allEntropies(pulseIdx + (iter-1)*numIterations));
        
    end 

    % save an image for each iteration 
    % img_iter = sum(scan_iter,4);
    % if iter == numIterations
    %     filename = sprintf('finalImage');
    % else 
    %     filename = sprintf('focusedImageIter_%d', iter);
    % end
    % save(filename, 'img_iter');

end

% % save the final pulse set
% save('focusedImageFinal.mat', 'scan_iter');
focused_image = scan_iter;

end
 
function [entropy] = findEntropy(img)

voxelInt = img .* conj(img);         
imgEnergy = findEz(voxelInt);                   
imgNormInt = voxelInt / imgEnergy;
entropy = - sum(sum(sum(imgNormInt.*log(imgNormInt))));

end

function [Ez] = findEz(img_int)

Ez = sum(sum(sum(img_int)));

end

        

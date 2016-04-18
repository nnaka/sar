%Entropy Minimizing Autofocus Algorithm
%  Colin Watts and Ian Fletcher
%  
%  This imploments an entropy minimizing autofocus algorithm to focus an
%  image from a set of radar pulse returns. 

%TODO: Remove constants from visualization
%      Create metric for when to end autofocus
%      Replace fminsearch with Coordinate Decent algorithm 

function [ Entropy ] = autoFocus(  )
    % Set up for visualization
    % Constants are subject to change due to changes in image size
    sceneSizeX = 100;
    sceneSizeY = 100;
    imgX = linspace(-sceneSizeX/2,sceneSizeX/2,641).';
    imgY = linspace(0,sceneSizeY,481).';
    myImg = zeros(numel(imgY),numel(imgX));
    hFig = figure;
    hAx = axes;
    hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
    colormap(hAx,jet(256));
    colorbar;
    xlabel('X (meters)');
    ylabel('Y (meters)');
    set(hAx,'YDir','normal');

    % Set of radar returns to be focused are loaded from file 'pulseSet.mat' 
    load('pulseSet.mat');
    
    % Initial values are calculated 
    imgSize = size(pulseSet(:,:,1));
    numPulses = length(pulseSet(1,1,:));
    myImg = sum(pulseSet,3);
    original = myImg;
    Entropy = FindEntropy(myImg);
    phi = zeros(numPulses,1);
    focusedPulses = pulseSet;
    
    % Loops through pulses and focues image
    num_iterations = 1;
    for iter = 1:num_iterations %TODO: come up with metric of when to stop iterating 
       for k = 1:numPulses
           % minPhi is  an anonomous function that is used in fminsearch to
           % find the phi offset required to minimize Entropy
           minPhi = @(x) FindMinEntropy(pulseSet(:,:,k)*exp(1j*x), focusedPulses, k);
           phi(k) = fminsearch(minPhi,phi(k));
           
           % Update values 
           focusedPulses(:,:,k) = pulseSet(:,:,k)*exp(1j*phi(k));
           myImg = sum(focusedPulses,3);
           Entropy = FindEntropy(myImg);
           E(k+(iter-1)*numPulses) = Entropy;
           
           % Update Image 
           img_dB = 20*log10(abs(myImg));
           set(hImg,'CData',img_dB);
           set(hFig,'Name',sprintf('Pulse %i of %i in iteration %i of %i. Current Entropy: %f', k, numPulses, iter, num_iterations,Entropy));
           if ~isinf(max(img_dB(:)))
               caxis(hAx,max(img_dB(:)) + [-20 0]);
           end
           drawnow;  
       end
    end
end

% Helper function for minPhi
function [Entropy] = FindMinEntropy(currentPulse,focusedPulses,k)
    focusedPulses(:,:,k) = currentPulse; 
    Entropy = FindEntropy(sum(focusedPulses,3));
end

% Calculates Entropy of an image
function [Entropy] = FindEntropy(myImg)
	pixInt = myImg.*conj(myImg);
	Ez = sum(sum(pixInt));
	normPix = pixInt/Ez;
	Entropy = -sum(sum(normPix.*log(normPix)));
end





% USED TO FIND AND PLOT ENTROPY VS PHI FOR FIRST PULSE

%     min = Inf;
%     minIdx = 0;
%     for i = 1:length(phiset)
%         phi(1) = phiset(i);
%         focusedPulses(:,:,1) = pulseSet(:,:,1)*exp(1j*phi(1));
%         myImg = sum(focusedPulses,3);
%         pixInt = myImg.*conj(myImg);
%         Ez = sum(sum(pixInt));
%         normPix = pixInt/Ez;
%         Entropy = -sum(sum(normPix.*log(normPix)));
%         test(i) = Entropy;
%         if test(i) < min
%             min = test(i);
%             minIdx = i;
%         end
%     end
%     
%     plot(phiset,test)
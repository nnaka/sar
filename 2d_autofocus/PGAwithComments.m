% PGA.m
%   An implomentation of a phase gradient autofocus algorithm. This is done
%   in four steps
%       1. Centering the brightest points in the image
%       2. Windowing the brightest points
%       3. Estimaging the phase gradient
%       4. Iteration
%   There are some issues with this algorithm that are not entirely clear
%   right now. My guesses at the source of the error is threefold
%       1. Data at a specific range in the image curves so when processing
%       the image information is lost or observed to be in a place where it
%       sould not be
%       2. Windowing is not done properly. I think that windowing may be ok
%       but I would like to look at it closer
%       3. The metric for iteration may be incorrect

function [ ] = PGA( )
    % Set up figure for image formation
    sceneSizeX = 100;
    sceneSizeY = 100;
    imgX = linspace(-sceneSizeX/2,sceneSizeX/2,641).';
    imgY = linspace(0,sceneSizeY,481).';
    myImg = zeros(numel(imgY),numel(imgX));
    hFig = figure;
    hAx = axes;
    hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
    currentPulse = zeros(size(myImg));
    colormap(hAx,jet(256));
    colorbar;
    xlabel('X (meters)');
    ylabel('Y (meters)');
    set(hAx,'YDir','normal');
    

    % Load image from file and save size
    load('MyImg.mat');
    imgSize = size(myImg);

    % RMS started at an arbitrary value > .1
    RMS = 10;
    j = 1;

    % This is where the iteration metric is checked
    %while RMS > .1
    while j < 100
        % Initialization
        centeredImg = zeros(imgSize);
        phi = zeros(1,imgSize(2));
        
        % 1: Center brightest points of image
        [~, maxIdx] = max(myImg,[],2);
        midpoint = ceil(imgSize(2)/2);

        for i = 1:imgSize(1)
            centeredImg(i,:) = circshift(myImg(i,:), midpoint-maxIdx(i),2);
        end
        
        
        % 2: Window Image
        centMag = centeredImg .* conj(centeredImg);
        Sx = sum(centMag, 1);
        Sx_dB = 20*log10(abs(Sx));
        cutoff = max(Sx_dB) - 10;

        W = 0;
        WinBool = Sx_dB >= cutoff;
        W = sum(WinBool);
        
        %Two windows have been tested, a normal curve and a square window
        x = 1:length(Sx);
        W = W*1.5;
        window = x > (midpoint - W/2) & x < (midpoint + W/2);
        %window = exp(-(x-midpoint).^2/(2*(W)^2));
        window = kron(window, ones(imgSize(1),1));
        windowedImg = centeredImg.*window; 

        % 3. Gradient Generation done by 2 methods
        
        %Minimum Varience
        Gn = ifft(windowedImg,[],2);
        dGn = ifft(1j*kron(x-midpoint,ones(imgSize(1),1)).*windowedImg,[],2);

        num = sum(imag(conj(Gn).*dGn),1);
        denom = sum(conj(Gn).*Gn,1);

        dPhi = num./denom;
        %Maximum Likelihood
        dPhi2 = zeros(length(dPhi),1);
        for k = 2:length(phi) 
            dPhi2(k) = (sum(Gn(:,k).*conj(Gn(:,k-1))));
            dPhi2(k) = dPhi2(k)/imgSize(1);
            dPhi2(k) = atan(imag(dPhi2(k))/real(dPhi2(k)))+pi/2*sin(imag(dPhi2(k)))*(1-sin(real(dPhi2(k))));
        end
        % Integration of phase gradients to find phase offset
        for i = 1:length(phi)
            phi(i) = sum(dPhi(1:i));
            phi2(i) = sum(dPhi2(1:i));
        end
        %The Phase error functions found in each method appear to be very
        %simmilar. The only major is that the Minimum Varience method
        %seems to have a much greater magnitude so I've been scaling it to
        %the size of the Maximum Likelihood estimation for comparision 
        phi = max(abs(phi2))*phi/(max(abs(phi)));

        %Add Phase difference estimation to current image and update image
        change = exp(-1j*phi2);
        myImg = fft(ifft(myImg,[],2).*kron(ones(481,1),change),[],2);
        
        % find RMS value for removed phase. To be used for iteration
        RMS = rms(phi2);

        % Update Image
        img_dB = 20*log10(abs(myImg));
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf('PGA Iteration: %i',j));
        if ~isinf(max(img_dB(:)))
            caxis(hAx,max(img_dB(:)) + [-20 0]);
        end
        drawnow;

        j = j+1;

    end
end


function [ ] = PGA( )
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
    

%     load('pulseSet.mat');
%     myImg = sum(pulseSet,3);
 load('myimg.mat');
%     myImg = TranslateImg(sum(pulseSet,3));
    imgSize = size(myImg);

    load('testq2.mat');
    corruptedImg = fft((ifft(myImg')'.*kron(ones(481,1),exp(1j*phioffset)))')';
%       myImg = corruptedImg;

    RMS = 10;
    j = 1;

    while RMS > .1
    % for j = 1:100

        centeredImg = zeros(imgSize);

        testphi = zeros(imgSize(2),1);

        [~, maxIdx] = max(myImg,[],2);
        midpoint = ceil(imgSize(2)/2);

        for i = 1:imgSize(1)
            centeredImg(i,:) = circshift(myImg(i,:), midpoint-maxIdx(i),2);
        end

        centMag = centeredImg .* conj(centeredImg);
        Sx = sum(centMag, 1);
        Sx_dB = 20*log10(abs(Sx));
        cutoff = max(Sx_dB) - 10;

        W = 0;
        WinBool = Sx_dB >= cutoff;
        W = sum(WinBool);
        
        %add window here
        x = 1:length(Sx);
        %window = exp(-(x-midpoint).^2/(2*(W)^2));
        W = W*2;
        window = x > (midpoint - W/2) & x < (midpoint + W/2);
        window = kron(window, ones(imgSize(1),1));
        windowedImg = centeredImg.*window; 

        Gn = ifft(windowedImg,[],2);
        dGn = ifft(1j*kron(x-midpoint,ones(imgSize(1),1)).*windowedImg,[],2);


        num = sum(imag(conj(Gn).*dGn),1);
        denom = sum(conj(Gn).*Gn,1);

        test = num./denom;
        test2 = zeros(length(test),1);
        for k = 2:length(testphi) 
            test2(k) = (sum(Gn(:,k).*conj(Gn(:,k-1))));
            test2(k) = test2(k)/imgSize(1);
            test2(k) = atan(imag(test2(k))/real(test2(k)))+pi/2*sin(imag(test2(k)))*(1-sin(real(test2(k))));
        end
        for i = 1:length(testphi)
    %         testphi(i) = sum(test2(1:i));
    %        testphi(i) = sum(test(1:i)/max(test)*max(-test2)); 
    %         testphi2(i) = sum(test(1:i)/imgSize(1));
    %        testphi2(i) = sum(1.5*test(1:i)/max(test)*max(-test2)); 

            testphi(i) = sum(test(1:i));
            testphi2(i) = sum(test2(1:i));
        end
        testphi = max(abs(testphi2))*testphi/(max(abs(testphi)));
        phiSet(j,:) = testphi;
        phiSet2(j,:) = testphi2;
        testphi = testphi';

        change = exp(-1j*testphi);
        prevImg = myImg;
        myImg = fft(ifft(myImg,[],2).*kron(ones(481,1),change),[],2);
        q = sum(phiSet,1);
        qq = sum(phiSet2,1);
        RMS = rms(testphi);

        img_dB = 20*log10(abs(myImg));
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf('PGA Iteration: %i',j));
        if ~isinf(max(img_dB(:)))
            caxis(hAx,max(img_dB(:)) + [-50 0]);
        end

        rmsvec(j) = RMS;
        fprintf('RMS: %f\n',RMS);

%        plot(x,testphi,x,testphi2)
%         plot(rmsvec)

        drawnow;

        %fprintf('End of iteration %i\n',q);
        j = j+1;

    end
    % plot(x,phioffset,x,qq)
end


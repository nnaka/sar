function [] = mip(img)

% ----------------------------------------------------------------------- %
%
% ViewCube
%
% Display processing image data as a 3D isosurface
%
% Inputs:
%   img - a 3D image cube or 4D set of pulses and their correspoding images
%   dB  - the number of dBs from the maximum that can form the isosurface
%
% ----------------------------------------------------------------------- %

myCube = sum(img, 4);

cube_dB = 20*log10(abs(myCube));
    sceneSizeX = 100;
    sceneSizeY = 100;
    imgX = linspace(-sceneSizeX/2,sceneSizeX/2,50).';
    imgY = linspace(0,sceneSizeY,50).';

    rows = [5, 10,15];

    for i = 1:50
        for j = 1:50
           MIP(i,j,1) = max(myCube(:,i,j));
           MIP(j,i,2) = max(myCube(i,:,j));
           MIP(i,j,3) = max(myCube(i,j,:));
        end
    end

    for i = 1:3
       myImg = zeros(numel(imgY),numel(imgX));
        hFig = figure;
        hAx = axes;

        colormap(hAx,gray);
        colorbar;

        if i == 1
            hImg = imagesc(imgX,imgY-50,nan*myImg,'Parent',hAx);
            title('Front view maximum intensity plot');
            xlabel('X (meters)');
            ylabel('Z (meters)');
        end
        if i == 2
            hImg = imagesc(imgX+50,imgY-50,nan*myImg,'Parent',hAx);
            title('Side view maximum intensity plot');
            xlabel('Y (meters)');
            ylabel('Z (meters)');
        end
        if i == 3
            hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
            title('Top view maximum intensity plot');
            xlabel('X (meters)');
            ylabel('Y (meters)');
        end

        set(hAx,'YDir','normal');

        myImg = MIP(:,:,i);

        img_dB = 20*log10(abs(myImg));
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf(' '));

        if ~isinf(max(img_dB(:)))
            caxis(hAx,[max(img_dB(:))-20 max(img_dB(:))]);
        end

        max(img_dB(:));
        min(img_dB(:));

        drawnow;
    end
end

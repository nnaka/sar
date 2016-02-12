function [] = ViewCube2(img, dB)

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

myCube = sum(img,4);

cube_dB = 20*log10(abs(myCube));
% cubeSize = size(myCube);
% [X,Y,Z] = meshgrid(1:cubeSize(1),1:cubeSize(2),1:cubeSize(3));
% f = figure;
% p1 = patch(isosurface(X,Y,Z,cube_dB,max(cube_dB(:))-dB));
% max(cube_dB(:))
% p1 = patch(isosurface(X,Y,Z,cube_dB,-96.4567-dB));
% isonormals(X,Y,Z,cube_dB,p1);
% p1.FaceColor = 'red';
% p1.EdgeColor = 'none';
% daspect([1,1,1])
% 
% view(3); axis tight
% grid on
% camlight 
% lighting gouraud
% zlabel('Z axis')
% xlabel('X axis')
% ylabel('Y axis')
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
%         myImg = zeros(numel(imgY),numel(imgX));
%         hFig = figure;
%         hAx = axes;
%         hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
%         colormap(hAx,gray);
%         colorbar;
%         xlabel('X (meters)');
%         ylabel('Y (meters)');
%         set(hAx,'YDir','normal');
%         
%         myImg = squeeze(myCube(i,:,:));
%         
%         img_dB = 20*log10(abs(myImg));
%         set(hImg,'CData',img_dB);
%         set(hFig,'Name',sprintf(' '));
%         if ~isinf(max(img_dB(:)))
%             caxis(hAx,max(img_dB(:)) + [-20 0]);
%         end
%         drawnow;
    end
    for i = 1:1
       myImg = zeros(numel(imgY),numel(imgX));
        hFig = figure;
        hAx = axes;
        if i == 1
            hImg = imagesc(imgX,imgY-50,nan*myImg,'Parent',hAx);
        end
        if i == 2
            hImg = imagesc(imgX+50,imgY-50,nan*myImg,'Parent',hAx);
        end
        if i == 3
            hImg = imagesc(imgX,imgY,nan*myImg,'Parent',hAx);
        end
        colormap(hAx,gray);
        colorbar;
        xlabel('X (meters)');
        ylabel('Z (meters)');
        set(hAx,'YDir','normal');
        
        myImg = MIP(:,:,i);
        
%         img_dB = 20*log10(abs(myImg));
        img_dB = abs(myImg);
        set(hImg,'CData',img_dB);
        set(hFig,'Name',sprintf(' '));
        if ~isinf(max(img_dB(:)))
%             caxis(hAx,max(img_dB(:)) + [-23 0]);
            caxis(hAx,[2.2125e-08 1.4899e-05]);
        end
        max(img_dB(:))
        min(img_dB(:))
        drawnow; 
        
    end
end
function [] = ViewCube(img, dB)

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
cubeSize = size(myCube);
[X,Y,Z] = meshgrid(1:cubeSize(1),1:cubeSize(2),1:cubeSize(3));
f = figure;
p1 = patch(isosurface(X,Y,Z,cube_dB,max(cube_dB(:))-dB));

isonormals(X,Y,Z,cube_dB,p1);
p1.FaceColor = 'red';
p1.EdgeColor = 'none';
daspect([1,1,1])

view(3); axis tight
grid on
camlight 
lighting gouraud
zlabel('Z axis')
xlabel('X axis')
ylabel('Y axis')

end
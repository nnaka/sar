% -----------------------------------------------------------------------------
% Display image data as a 3D isosurface
% 
% @param img [Array] an X by Y by Z 3D image
% @param dB  [Integer] the number of dBs from the maximum that can form the
% isosurface
% -----------------------------------------------------------------------------
function view3DImage( image, dB )

image = mag2db(abs(image));

cubeSize = size(image);
[X,Y,Z] = meshgrid(1:cubeSize(1), 1:cubeSize(2), 1:cubeSize(3));

figure;

p1 = patch(isosurface(X, Y, Z, image, max(image(:)) - dB));
isonormals(X, Y, Z, image, p1);

p1.FaceColor = 'red';
p1.EdgeColor = 'none';
daspect([1,1,1])

view(3);
axis tight
grid on
camlight 
lighting gouraud
xlabel('X (meters)')
ylabel('Y (meters)')
zlabel('Z (meters)')

end

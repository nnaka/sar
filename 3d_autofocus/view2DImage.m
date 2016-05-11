% -----------------------------------------------------------------------------
% Display 2D image data
% 
% @param img [Array] an X by Y 2D image
% -----------------------------------------------------------------------------
function view2DImage( image )

figure;
hAx = axes;

imagesc(mag2db(abs(image)), 'Parent', hAx);

colormap(hAx,gray);
colorbar;
xlabel('X (meters)');
ylabel('Y (meters)');
set(hAx,'YDir','normal');

end

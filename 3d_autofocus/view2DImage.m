% -----------------------------------------------------------------------------
% Display 2D image data
% 
% @param img [Array] an X by Y 2D image
% -----------------------------------------------------------------------------
function view2DImage( image )

figure;
hAx = axes;

image = mag2db(abs(image));

hImg = imagesc(image, 'Parent', hAx);

colormap(hAx, gray);
colorbar;
set(hAx,'YDir','normal');

% if ~isinf(max(image(:)))
%   caxis(hAx, max(image(:)) + [-20 0]);
% end

end

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

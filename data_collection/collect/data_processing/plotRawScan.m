function [] = plotRawScan(rawscan, scan_dim, scanResPs)
% Plot the raw scan data incrimentally

C_mps = 299792458;

figure;
for i = 1:scan_dim(1)
    distance = ([0:length(rawscan(i,:))-1]*scanResPs/1e12)*C_mps/2;
    
    hold on
    plot(distance, rawscan(i,:))
    xlabel('Distance (m)');
    grid on;
    drawnow;
    
    pause(0.2);
end

end 
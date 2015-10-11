% pltoGPS.m
% By Colin Watts and Ian Fletcher
% 
%   This code is designed to take GPS Data and, with a little bit of human
%   input, plot the movement of the GPS and then calculate statistics based
%   on the motion of the GPS in a straight line
%
%   If the statistics are not needed or do not fit the appropriate data set
%   feel free to reuse and modify as needed
%
%   A few user inputs are required for each data set in order to generate
%   appropriate statistics


function [] = plotGPS()

    text = fileread('csv/baseline_log_linepullWvid.csv');

    % I have found that strsplit() does some weird things to the variable
    % so there are some weird lines that are required for parsing.
    C = strsplit(text);
    
    hFig = figure;
    
    count = 1;
    j = 1;

    
%THIS IS A LIST OF ALL OF THE FILE SPECIFIC DATA. MAKE SURE THIS FITS WITH YOUR CHOSEN FILE    
    %NSpull1
%     cutoff = 1;
%     data_start = 800;
%     data_length = 150;
    
    % For test 4
%     cutoff = 100;
%     data_start = 619; 
%     data_length = 100;

    % for line pull with video
    cutoff = 100;
    data_start = 345;
    data_length = 41;

    % for test 2
%     cutoff = 200;
%     data_start = 1400;
%     data_length = 280;

    for i = 2:length(C)
        str = strsplit(char(C(i)),',');
        if (length(str) == 8)
            GPS_Data(count).time = str(1);
            GPS_Data(count).north = str2double(char(str(2)));
            GPS_Data(count).east = str2double(char(str(3)));
            GPS_Data(count).down = str2double(char(str(4)));
            GPS_Data(count).distance = str2double(char(str(5)));
            GPS_Data(count).num_sats = str(6);
            GPS_Data(count).flags = char(str(7));
            GPS_Data(count).num_hypothesis = str(8);
            
            time = strsplit(char(GPS_Data(count).time),':');

            % This ensures that only data taken in RTK lock is looked at
            % (also ensures that it is within 200 m as the RTK lock can be difficult) 
            if  strcmp(GPS_Data(count).flags,'0x01') && GPS_Data(count).distance < 200
                X(j) = GPS_Data(count).east;
                Y(j) = GPS_Data(count).north;
                Z(j) = GPS_Data(count).down;
                D(j) = GPS_Data(count).distance;
                
                %Update plot with new point
                scatter(X,Y);
                drawnow;
                j = j+1;
            end
            set(hFig,'Name',sprintf('Time: %s', char(GPS_Data(count).time)));
            count = count+1;
        end
    end
% End of plotting section
    
% Beginning of statistics section

    % beginning and end of Data is typically not good data
    x = X(cutoff:length(X)-cutoff);
    y = Y(cutoff:length(Y)-cutoff);
    z = Z(cutoff:length(Z)-cutoff);
    d = D(cutoff:length(D)-cutoff);
    
    % finds the the mean position of stationary data at the beginning of
    % collect
    xMean1 = mean(x(1:data_start-50));
    yMean1 = mean(y(1:data_start-50));
    zMean1 = mean(z(1:data_start-50));
    dMean1 = mean(d(1:data_start-50));
    
    % finds the the mean position of stationary data at the end of collect
    xMean2 = mean(x(length(x)-150:end));
    yMean2 = mean(y(length(y)-150:end));
    zMean2 = mean(z(length(z)-150:end));
    dMean2 = mean(d(length(d)-150:end));
    
    % used for drawing interpolated line on plot
    qx = linspace(xMean1,xMean2, 5000);
    qy = linspace(yMean1,yMean2, 5000);
    
    % Plots pruned data on grid with interpolated line
    scatter(x,y);
    hold on
    scatter([xMean1,xMean2],[yMean1,yMean2],40,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
    plot(qx, qy)
    hold off
    

    origin = [xMean1, yMean1, 0];
    vec = [xMean2, yMean2, 0] - origin;
    uVec = vec/sqrt(vec(1)^2 + vec(2)^2);
  
    % finds the error in the y direction. The y direction is orthagonal to path
    % of the aperture in the NE plance
    for i = 1:data_length
        pt = [x(i+data_start), y(i+data_start), 0]-origin;
        proj(i) = dot(pt,uVec);
        temp = pt-uVec*proj(i);
        if temp(1) < temp(2)
            yd(i) = sqrt(temp(1)^2+temp(2)^2);
        else
            yd(i) = -sqrt(temp(1)^2+temp(2)^2);
        end
    end
    
    origin = [xMean1, yMean1, zMean1];
    vec = [xMean2, yMean2, zMean2] - origin;
    uVec = vec/sqrt(vec(1)^2 + vec(2)^2 + vec(3)^2);
    
    % now we put everything in 3D and find the error in the Z (orthagonal to
    % both the Y and the X directions
    for i = 1:data_length
        pt = [x(i+data_start), y(i+data_start), z(i+data_start)]-origin;
        proj2(i) = dot(pt,uVec);
        temp = pt-uVec*proj2(i);
        zd(i) = temp(3);
    end    
    
end


% ----------------------------------------------------------------------- %
% Cut Image Cube
%
% This script takes a 4D pulse set and returns a subsection of the set as
% defined by the axe bounds x,y,z start and end. The goal is to decrease
% the autofocus processing time by only focusing the voxels in the image
% that are of immediate relevance. 
%
% ----------------------------------------------------------------------- %


scanSet = pulseSet;
 
xStart = 25;                    % Define the starting and ending positions
xEnd = 30;                      % for each dimension.
xLength = xEnd - xStart;        
                                % Best practice is to cut the image into a 
yStart = 25;                    % smaller cube, rather than a rectangle. 
yEnd = 30;                      % There can be some issues with visualizing
yLength = yEnd - yStart;        % and processing rectangular data sets.

zStart = 25;
zEnd = 30;
zLength = zEnd - zStart;

numScans = size(scanSet,4);
newImg = zeros(yLength, xLength, zLength, numScans);

for x = 1:xLength
    for y = 1:yLength
        for z = 1:zLength
            newImg(y,x,z,:) = scanSet(y + yStart - 1,x + xStart - 1,z + zStart - 1,:);
        end
    end
end

cutImg = newImg;
save cutImage.mat cutImg -v7.3
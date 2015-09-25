% ----------------------------------------------------------------------- %
% Form and Process Image
% 
% 1 - Generates a set of radar point reflectors 
% 2 - Forms pulsed radar data from these points
% 3 - Processes the radar data to form a 3D image
% 4 - Autofocus the image if the pulse set has the proper number of
%     dimensions* 
% 5 - View and save the image
%
%
% *Autofocus requires a 4D matrix: the 1D for each pulse and 3D for each
%  corresponding image. SAR 3D will output a 4D matrix if specified with 
%  the boolean form_pulse_set. 
% Current Limitations:
%   If you are trying to create a pulseSet to run with autofocus with an
%   aperture size greater than 40x40 pulses and image size greater than
%   50x50x50 voxels , MATLAB will consume every bit of RAM and implode. 
%   Intensive memory use is a HUGE issue here. Unless your computer is a 
%   computational beast with limitless RAM, it will run out of storage. 
%
%   If you are thinking, wait, a 40x40 pulse aperture is many orders 
%   magnitude smaller than the ideal aperture size, you are correct. This 
%   memory problem will grow exponentially as we increase aperture sizes to
%   our operational goal of 30x5m with ~110,000 pulses. 
%
%   Several options going forward: 
%       1 - Redesign autofocus to work with 3D data sets. 
%       2 - Partition the 4D pulse sets into numerous smaller ones. This
%           way data is saved incrimentally and is not all stored at once.
%           It can be incrimentally recombined as needed in the autofocus
%           routine. 
%       3 - Build a computational beast with limitless RAM.
%
% Also of note:
%   With current numbers for GPS error (std X, Y = 0.0051m; Z = 0.0071m)
%   GPS error is unnoticable in images created from apertures upwards of
%   100x100 pulses, and barely noticable in apertures around 40x40 pulses. 
% ----------------------------------------------------------------------- %

%% Generate target points 
% these vectors define the locations of the set of point source scatterers 
% that we model in the scenario. 
% so long as each point has an [X Y Z] coordinate, there can be any number
% of points in any arrangement that you choose...have fun!

 x0=kron((-20:20:20)',ones(3,1));     
 y0=kron(ones(3,1),(-20:20:20)');
 z0=[0;30;-30;30;0;-30;30;-30;0];

% x0 = linspace(-5,5,100)';
% y0 = zeros(numel(x0),1);
% z0 = y0;

% x0 = [-10,0,10];
% y0 = [-10,0,10];
% z0 = [-10,0,10];

%x0 = [-30,-30,-30,-30,-30,-20,-10,-10,-10,-10,-10,10,20,30,20,20,20,10,20,30];
%y0 = [10,5,0,-5,-10,0,10,5,0,-5,-10,10,10,10,5,0,-5,-10,-10,-10];
%z0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'

% Other image parameters
nX = 30;                       % # of points per row in synthetic aperture
nY = 30;                       % # of rows in synthetic aperture
addNoise = true;               % true will inject GPS noise into aperture

%% Generate SAR data with the given points 
PulseData = FormPulses(x0, y0, z0, nX, nY, addNoise);

%% Format the raw radar data to mesh with 2014-2015 Senior Design code
formedData = FormatPulseData(PulseData);

%% Process formed data to create an image
imgSize = [50 50 50];        % voxels [ X Y Z ]
sceneSize = [100 100 100];      % meters [ X Y Z ] 
form_pulse_set = true;         % set to true if image is to be autofocused
imageSet = SAR_3D(formedData, imgSize, sceneSize, form_pulse_set);

%% Perform minimum entropy Autofocus 

% The autofocus algorithm will only run with a 4D data set 
% if the given data set is 3D, the image will not be autofocused
if numel(size(imageSet)) == 4                 
    num_iter = 1;           % number of iterations to run autofocus
    final_image = minEntropyAutoFocus(imageSet, num_iter);
else 
    final_image = imageSet;
end 

%% Display the final image

dB = 10;
ViewCube(final_image, dB);

save imageCube.mat final_image -v7.3



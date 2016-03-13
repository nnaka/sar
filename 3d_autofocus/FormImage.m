% ----------------------------------------------------------------------- %
% Form Image
% 
% Forms an image using the parameters set below and saves it to the specified
% filename
%
% 1 - Generates a set of radar point reflectors 
% 2 - Forms pulsed radar data from these points
% 3 - Processes the radar data to form a 3D image
%
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
% Also of note:
%   With current numbers for GPS error (std X, Y = 0.0051m; Z = 0.0071m)
%   GPS error is unnoticable in images created from apertures upwards of
%   100x100 pulses, and barely noticable in apertures around 40x40 pulses. 
% ----------------------------------------------------------------------- %

% @param filename [string] e.g. '/data/imageSet50by50_9pts_noise(0,0,0)'
function [] = FormImage( filename )

%% Generate target points 
% these vectors define the locations of the set of point source scatterers 
% that we model in the scenario. 
% so long as each point has an [X Y Z] coordinate, there can be any number
% of points in any arrangement that you choose...have fun!

% x0=kron((-20:20:20)',ones(3,1));     
% y0=kron(ones(3,1),(-20:20:20)');
% z0=[0;30;-30;30;0;-30;30;-30;0];

x0 = 10;
y0 = 10;
z0 = 10;

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
nX = 35;                       % # of points per row in synthetic aperture
nY = 35;                       % # of rows in synthetic aperture
noiseVec = [0,0,0];          % nonzero values will inject noise

%% Generate SAR data with the given points 
PulseData = FormPulses(x0, y0, z0, nX, nY, noiseVec);

% For testing purposes, we do not generate new data each run, and instead
% load formatted data from a file.

%% Format the raw radar data to mesh with 2014-2015 Senior Design code
formedData = FormatPulseData(PulseData);

% For testing purposes, we load imageSet strictly from a file so it is
% consistent


%% Process formed data to create an image
imgSize = [50 50 50];        % voxels [ X Y Z ]
sceneSize = [100 100 100];   % meters [ X Y Z ] 
imageSet = SAR_3D(formedData, imgSize, sceneSize, true);

% For testing purposes, we load imageSet strictly from a file so it is
% consistent.

save(filename, 'imageSet', '-v7.3')

end

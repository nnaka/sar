% IRDataImport
% Import data from IR sensor csv output file; outputSize
% determines number of rows in output data. Output data has
% four columns - time, x position, y position, and z position.
% Enter outputSize of 0 for uncompressed original IR data.
function data = IRDataImport(fileName, outputSize)
	rowNum = 1;			% initialize starting row
	rawDataSize = 0;	% initialize # rows of raw data
	t = [];
	x = [];
	y = [];
	z = [];

	fid=fopen(fileName);

	while ~feof(fid)					% iterate through all rows
		l = fgetl(fid);					
		if rowNum>50 && ~mod(rowNum-51,4)	% access row elements if row greater than 45 and even
			c = strsplit(l,',');		% parse row elements by commas and store in string vector
			t = [t; str2double(c(3))];	% convert necessary elements to double and store in vector
			x = [x; str2double(c(8))];
			y = [y; str2double(c(9))];
			z = [z; str2double(c(10))];
			rawDataSize = rawDataSize+1;% count # rows of raw data
		end
		rowNum = rowNum+1;				% count # rows total
	end

	fclose(fid);

	plot3(x,y,z)
	rawData = [t,x,y,z];	% concatenate data into matrix

	if outputSize>0									% if outputSize ~0
		outputSize = outputSize-1;					% account for inital row
		data = rawData(1,:);						% store first row of rawData to data
		for n = 1:outputSize
			a = round((n*rawDataSize)/outputSize);	% create intervals to match specified outputSize
			data = vertcat(data, rawData(a,:));		% concatenate intervals of rawData to data
		end
	else
		data = rawData;		% if outputSize 0 use all data
	end


end

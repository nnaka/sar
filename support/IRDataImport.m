% IRDataImport
% Import data from IR sensor csv output file; outputSize
% determines number of rows in output data. Output data has
% four columns - time, x position, y position, and z position.
function positionData = IRDataImport(fileName)
	rowNum = 1;			% initialize starting row
	rawDataSize = 0;	% initialize # rows of raw data
	t = [];
	x = [];
	y = [];
	z = [];

	fid=fopen(fileName);

	while ~feof(fid)					% iterate through all rows
		l = fgetl(fid);					
		if rowNum>50 && ~mod(rowNum-61,9)	% access row elements if row greater than 45 and even
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

	% figure(5);
	% plot3(x,y,z)
	% grid on;
	positionData = [t,x,y,z];	% concatenate data into matrix

end

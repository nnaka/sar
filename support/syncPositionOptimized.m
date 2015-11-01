%% rawCollect
% Matches IR position data to radar returns based on time since the
% beginning of the data collection.  Search is optimized for radar w/
% ~1sec/scan and IR collection w/ ~60fps.  Search range examples are
% as follows: 1-100 frames (Scan 1), 60-160 frames (Scan 2), and
% 140-240 frames (Scan 3).

function rawCollect = syncPosition(rawCollect, positionData)
	range = 0;		% Range of positionData values to search
	e=0.01;			% Error

	% Iterate through scans
	for i=1:length(rawCollect)

		% Round scan time to tenth of a second
		rawTime = rawCollect{i}.time;

		% Iterate through 100 frames (scan~1sec, IR~60fps)
		for j=1:100

			index = (j+range)-20;

			% Set frame window to limit search and match scan time
			if i>1 & positionData(index)
				posTime = positionData(index,1);
			else
				index = j;
				posTime = positionData(index,1);
			end

			% If times match within error, store position
			if (posTime-e<=rawTime & rawTime<=posTime+e) 
				rawCollect{i}.xLoc_m = positionData(index,2);
				rawCollect{i}.yLoc_m = positionData(index,3);
				rawCollect{i}.zLoc_m = positionData(index,4);
				rawCollect{i}.indexMatch = index;

				figure(5)
				hold all;
				scatter3(rawCollect{i}.xLoc_m,rawCollect{i}.yLoc_m,rawCollect{i}.zLoc_m);
				drawnow;
				break;
			end

		end

		% Move to next time range (~1 second)
		range = range+80;

	end

	grid on;
end
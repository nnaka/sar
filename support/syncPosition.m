%% rawCollect
% Matches IR position data to radar returns based on time since the
% beginning of the data collection.

function rawCollect = syncPosition(rawCollect, positionData)
	e=0.01;			% Error
	index=1;		% Index in position data to speed search

	% Iterate through scans
	for i=1:length(rawCollect)

		% Round scan time to tenth of a second
		rawTime = rawCollect{i}.time;

		% Iterate through frames
		for j=index:length(positionData)

			posTime = positionData(j,1);

			% If times match within error, store position
			if (posTime-e<=rawTime && rawTime<=posTime+e) 
				rawCollect{i}.xLoc_m = positionData(j,2);
				rawCollect{i}.yLoc_m = positionData(j,3);
				rawCollect{i}.zLoc_m = positionData(j,4);
				% rawCollect{i}.indexMatch = j;
				index = j;

				figure(3)
				hold all;
				scatter3(rawCollect{i}.xLoc_m,rawCollect{i}.yLoc_m,rawCollect{i}.zLoc_m);
				drawnow;
				break;
			end

		end

	end

	grid on;
end
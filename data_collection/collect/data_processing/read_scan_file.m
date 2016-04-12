function [raw_scan, gps_data] = read_scan_file(fileName)

% every new line is read in as a zero row
rawData = dlmread(fileName, ',', 1, 1);
dim = size(rawData);

% eliminate rows with all zeros 
for i = 0:floor(dim(1)/2)
    data(i+1,:) = rawData(2*i + 1,:);
end 

data_size = size(data);

% separate radar data 
raw_scan = data(4:data_size(1),1:end-4);
gps_data = data(4:data_size(1),(end-3):end);

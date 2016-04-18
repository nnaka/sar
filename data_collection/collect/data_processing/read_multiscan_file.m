function [raw_scan, gps_data] = read_multiscan_file(file_name)
% reads a file with multiple scans to create a 3D radar scan array
%
% 1st Dimension: scan values and GPS data
% 2nd Dimension: every scan in the given row
% 3rd Dimension: every row in the data set


raw_data = str2double(read_mixed_csv(file_name, ','));
dim = size(raw_data);
num_rows = raw_data(dim(1), 1);
raw_data(dim(1), 1) = NaN;

% split the data by row
idx = 2;                % we know that row 2 is start of first row
scan_idx = 1;
row_idx = 1;
while row_idx <= num_rows
    
    raw_scan(scan_idx,:,row_idx) = raw_data(idx,1:end-3);
    gps_data(scan_idx,:,row_idx) = raw_data(idx,end-2:end);
    scan_idx = scan_idx + 1;
    idx = idx + 1;
    
    if isnan(raw_data(idx,1))
        row_idx = row_idx + 1;
        idx = idx + 1;
        scan_idx = 1;
    end 
   
end

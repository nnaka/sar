function binaryDat = struct2bin(matStr)

% MIT IAP 2013: Find a Needle in a Haystack
%
% Convert Matlab structure to binary data
%
% Inputs:
%   matStr: structure
%
% Outputs:
%   binaryDat: data
%
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

fieldNames = fieldnames(matStr);
numFields = length(fieldNames);

binaryDat = uint8([]);

for i = 1:numFields
	binaryDat = [binaryDat typecast(swapbytes(matStr.(fieldNames{i})),'uint8')];
end

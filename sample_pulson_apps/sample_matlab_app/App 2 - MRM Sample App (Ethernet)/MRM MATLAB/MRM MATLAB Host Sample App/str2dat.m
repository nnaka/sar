function dat = str2dat(str)
% STR2DAT Function to convert structure to data.

fld = fieldnames(str);
Nfld = length(fld);

dat = int8([]);

for n = 1:Nfld
  dat = [dat typecast(swapbytes(str.(fld{n})),'int8')];
end

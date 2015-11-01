function [cfg,req,scn,det] = readMrmRetLog(varargin)
% readMrmRetLog: Function to read MRM-RET log file.
%
% Syntax
% readMrmRetLog
% readMrmRetLog(fnm)
% readMrmRetLog(dnm,fnm)
%
% Input
% fnm - string containing file name or complete path name
% dnm - string containing directory name
%
% Output
% cfg - structure containing configuration data
% req - structure containing request data
% scn - structure containing scan data
% det - structure containing detection data
%
% Usage Notes
% If no file name is provided, readMrmRetLog opens a dialog for the user to
% select the desired file. The single input fnm can be a file name only or
% a complete path name. The two inputs dnm and fnm are combined to make a
% complete path name.
%
% See also UIGETFILE, FULLFILE.

% Copyright © 2011 Time Domain, Huntsville, AL


% Handle input arguments.
switch nargin
  case 0
    [fnm,dnm] = uigetfile('*.txt');
  case 1
    dnm = '';
    fnm = varargin{1};
  case 2
    dnm = varargin{1};
    fnm = varargin{2};
  otherwise
    error('Too many input arguments.')
end

% Open file.
fid = fopen(fullfile(dnm,fnm),'rt');

% Create empty structures to append when number of element counters exceeds
% size of structure. This process is used to allocate memory for structures
% in blocks as needed rather than adding one element at a time. Block size
% for appending to structures is hard coded. A modified version of this
% function with a larger value would be useful when reading extremely large
% files.
N = 100;

cfg_ = repmat(struct('T',[],'nodeID',[],'Tstrt',[],'Tstp',[],'Nbin',[],'BII',[],'seg1Nsamp',[],'seg2Nsamp',[],'seg3Nsamp',[],'seg4Nsamp',[],'seg1Iadd',[],'seg2Iadd',[],'seg3Iadd',[],'seg4Iadd',[],'Iant',[],'Gtmt',[],'Ichan',[]),1,N);
req_ = repmat(struct('T',[],'msgID',[],'Nscn',[],'Tint',[],'stat',[]),1,N);
scn_ = repmat(struct('T',[],'msgID',[],'srcID',[],'Tstmp',[],'Tstrt',[],'Tstp',[],'Nbin',[],'Nfilt',[],'antID',[],'Imode',[],'Nscn',[],'scn',[]),1,N);
%scn_ = repmat(struct('T',[],'msgID',[],'srcID',[],'Tstmp',[],'Qchan',[],'RSSI',[],'Oldedg',[],'Olckspt',[],'Tstrt',[],'Tstp',[],'Nbin',[],'Nfilt',[],'antID',[],'Imode',[],'Nscn',[],'scn',[]),1,N);
det_ = repmat(struct('T',[],'msgID',[],'Ndet',[],'det',[]),1,N);

% Initialize structures and number of element counters.
Kcfg = 0;
Kreq = 0;
Kscn = 0;
Kdet = 0;

cfg = [];
req = [];
scn = [];
det = [];

% Read file to end.
while ~feof(fid)
  % Get next line.
  ln = fgetl(fid);
  
  % Get first two fields. Second field is the log entry type.
  i = strfind(ln,',');
  fld = textscan(ln(1:i(2)-1),'%s %s','Delimiter',',');
  
  % Switch on log entry type. In each case, the number of elements counter
  % is incremented, the structure size is increased if necessary, fields
  % are extracted, and data are put in the associated structure fields.
  switch fld{1}{1}
    case 'Timestamp'
      % These are text lines describing the fields of each of the various
      % log entries.

    otherwise
      %fprintf('%s\n',ln)
      switch fld{2}{1}
        case 'Config'
          Kcfg = Kcfg + 1;
          if Kcfg > length(cfg)
            cfg = [cfg cfg_];
          end
          fld = textscan(ln,'%n %s %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n','Delimiter',',');
          cfg(Kcfg) = struct('T',fld{1},'nodeID',fld{3},'Tstrt',fld{4},'Tstp',fld{5},'Nbin',fld{6},'BII',fld{7},'seg1Nsamp',fld{8},'seg2Nsamp',fld{9},'seg3Nsamp',fld{10},'seg4Nsamp',fld{11},'seg1Iadd',fld{12},'seg2Iadd',fld{13},'seg3Iadd',fld{14},'seg4Iadd',fld{15},'Iant',fld{16},'Gtmt',fld{17},'Ichan',fld{18});

        case 'MrmControlRequest'
          Kreq = Kreq + 1;
          if Kreq > length(req)
            req = [req req_];
          end
          fld = textscan(ln,'%n %s %n %n %n','Delimiter',',');
          req(Kreq)= struct('T',fld{1},'msgID',fld{3},'Nscn',fld{4},'Tint',fld{5},'stat',nan);

        case 'MrmControlConfirm'
          % This log entry does not have its own structure. It is part of
          % the req structure and uses the latest req structure number of
          % elements counter.
          fld = textscan(ln,'%n %s %n %n','Delimiter',',');
          if fld{3} == req(Kreq).msgID
            req(Kreq).stat = fld{4};
          else
            error('MrmControlConfirm message ID does not match previous MrmControlRequest message ID.')
          end

        case 'MrmFullScanInfo'
          Kscn = Kscn + 1;
          if Kscn > length(scn)
            scn = [scn scn_];
          end
          i = strfind(ln,',');
          fld = textscan(ln(1:i(16)-1),'%n %s %n %n %n %n %n %n %n %n %n %n %n %n %n %n','Delimiter',',');
          scn(Kscn) = struct('T',fld{1},'msgID',fld{3},'srcID',fld{4},'Tstmp',fld{5},'Tstrt',fld{10},'Tstp',fld{11},'Nbin',fld{12},'Nfilt',fld{13},'antID',fld{14},'Imode',fld{15},'Nscn',fld{16},'scn',[]);
%          scn(Kscn) = struct('T',fld{1},'msgID',fld{3},'srcID',fld{4},'Tstmp',fld{5},'Qchan',fld{6},'RSSI',fld{7},'Oldedg',fld{8},'Olckspt',fld{9},'Tstrt',fld{10},'Tstp',fld{11},'Nbin',fld{12},'Nfilt',fld{13},'antID',fld{14},'Imode',fld{15},'Nscn',fld{16},'scn',[]);
          scn(Kscn).scn = str2num(ln(i(16)+1:end));

        case 'MrmDetectionListInfo'
          Kdet = Kdet + 1;
          if Kdet > length(det)
            det = [det det_];
          end
          i = strfind(ln,',');
          if length(i) < 4
            i(4) = length(ln) + 1;
          end
          fld = textscan(ln(1:i(4)-1),'%n %s %n %n','Delimiter',',');
          det(Kdet) = struct('T',fld{1},'msgID',fld{3},'Ndet',fld{4},'det',[]);
          if det(Kdet).Ndet > 0
            det(Kdet).det = reshape(str2num(ln(i(4)+1:end)),2,[]);
          end

      end
  end
end

% Close file.
fclose(fid);

% Trim structures arrays to elements actually filled.
cfg = cfg(1:Kcfg);
req = req(1:Kreq);
scn = scn(1:Kscn);
det = det(1:Kdet);

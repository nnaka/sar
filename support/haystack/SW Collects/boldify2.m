function boldify(h,g)
%BOLDIFY Make lines and text bold for standard viewgraph style.
%   BOLDIFY boldifies the lines and text of the current figure.
%   BOLDIFY(H) applies to the graphics handle H.
%   BOLDIFY(X,Y) specifies an X by Y inch graph of the current
%   figure.  If text labels have their 'UserData' data property
%   set to 'slope = ...', then the 'Rotation' property is set to
%   account for changes in the graph's aspect ratio.  The
%   default is MATLAB's default.
%

%               S. T. Smith
%
% The name of this function does not represent an endorsement by the author
% of the egregious grammatical trend of verbing nouns.
%
%  Submitted by Rob Haupt,  Maintained by Ray Quenneville

if nargin < 1, h = gcf;, end

% Set (and get) the default MATLAB paper size and position
set(gcf,'PaperPosition','default');
units = get(gcf,'PaperUnits');
set(gcf,'PaperUnits','inches');
fsize = get(gcf,'PaperPosition');
fsize = fsize(3:4);                     % Figure size (X" x Y") on paper.
psize = get(gcf,'PaperSize');

if nargin == 2                          % User specified graph size
  fsize = [h,g];
  h = gcf;
end

% Set the paper position of the current figure
set(gcf,'PaperPosition', ...
  [(psize(1)-fsize(1))/2 (psize(2)-fsize(2))/2 fsize(1) fsize(2)]);
fsize = get(gcf,'PaperPosition');
fsize = fsize(3:4);                     % Graph size (X" x Y") on paper.

set(gcf,'PaperUnits',units);            % Back to original

% Get the normalized axis position of the current axes
units = get(gca,'Units');
set(gca,'Units','normalized');
asize = get(gca,'Position');
asize = asize(3:4);
set(gca,'Units',units);

ha = get(h,'Children');

for i=1:length(ha)
  ha_type = get(ha(i),'Type');
  ha_tag  = get(ha(i),'Tag');
  if strcmp(ha_type,'axes') %& ~strcmp(ha_tag,'legend')
    %%%i  
    units = get(ha(i),'Units');
    set(ha(i),'Units','normalized');
    asize = get(ha(i),'Position'); % Axes Position (normalized)
    asize = asize(3:4);
    set(ha(i),'Units',units);

    [m,j] = max(asize); j = j(1);
    scale = 1/(asize(j)*fsize(j));      % scale*inches -> normalized units
    
    set(ha(i),'FontSize',16);           % Tick mark and frame format
    set(ha(i),'FontWeight','Bold');
    set(ha(i),'LineWidth',1);
    
    [m,k] = min(asize); k = k(1);
    if asize(k)*fsize(k) > 1/2
      set(ha(i),'TickLength',[1/8 2.5*1/8]*scale); % Gives 1/8" ticks
    else
      set(ha(i),'TickLength',[3/32 2.5*3/32]*scale); % Gives 3/32" ticks
    end

    set(get(ha(i),'XLabel'),'FontSize',16); % 14-pt labels
    set(get(ha(i),'XLabel'),'FontWeight','Bold');
    set(get(ha(i),'XLabel'),'VerticalAlignment','top');

    set(get(ha(i),'YLabel'),'FontSize',16); % 14-pt labels
    set(get(ha(i),'YLabel'),'FontWeight','Bold');
    set(get(ha(i),'YLabel'),'VerticalAlignment','baseline');

    set(get(ha(i),'Title'),'FontSize',16); % 16-pt titles
    set(get(ha(i),'Title'),'FontWeight','Bold');
  end
  hc = get(ha(i),'Children');
  if ~strcmp(ha_tag,'legend')
    for j=1:length(hc)
      chtype = get(hc(j),'Type');
      if chtype(1:4) == 'text'
        set(hc(j),'FontSize',16);         % 12 pt descriptive labels
        set(hc(j),'FontWeight','Bold');
        ud = get(hc(j),'UserData');       % User data
        if length(ud) > 8
          if ud(1:8) == 'slope = '        % Account for change in actual slope
            slope = sscanf(ud,'slope = %g');
            slope = slope*(fsize(2)/fsize(1))/(asize(2)/asize(1));
            set(hc(j),'Rotation',atan(slope)/pi*180);
          end
        end
      elseif chtype(1:4) == 'line'
        set(hc(j),'LineWidth',2);
      end
    end
  end
end


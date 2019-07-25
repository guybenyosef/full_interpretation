% -- Visualize image interpretation ---
% ---------------------------------------------------------------------------------
% Description: plot detailed image interpreation
% Type:        Visualization procedure
% INPUT:        figure_id: a natural number (refers to MATLAB's figure object for the plot)
%               img: an image in 108x108 pixel size. 
%               interpretation_set: a MATLAB strcut of points, contours, and regions
%               
%  Run, e.g.: 
%
% >> plot_image_interpretation(1,imresize(HORSE_HEAD(36).mirc,[108,108]),HORSE_HEAD(36).human_interpretation,true);
%
% Guy Ben-Yosef & Liav Assif, 2017, gby@csail.mit.edu
% ---------------------------------------------------------------------------------  %
function[] = plot_image_interpretation(figure_id,img,interpretation_set)
LWIDTH=3; 
  %LWIDTH=8; % for publications
  
show_img_flag = true;
if(show_img_flag)
    figure(figure_id); clf;imshow(img);
end

%        --- PLOTTING: --- :
contour_colors = {'r','c','y','g','m','b',[154, 185, 115]/256,[218, 218, 218]/256,[255, 192, 203]/256,[255, 239, 213]/256,[255, 127, 0]/256,[64, 224, 208]/256};  % light olive, light silver, pink, Peach, Orange, Turkiz
pts_colors = {'m','r','c','b','g','y'};
regions_colors = {[200, 165, 105]/256,[154, 185, 115]/256,[218, 218, 218]/256,'y','b','g','r'};

if(~isempty(interpretation_set))
    cls_pts = interpretation_set{1};
    cls_contours = interpretation_set{2};
    cls_regions = interpretation_set{3};
else
    cls_pts = {};   cls_contours = {}; cls_regions = {}; 
end
figure(figure_id);hold on;
primitive_names=[];
ph=[];


% Contours
for k=1:length(cls_contours)
    if(~isempty(cls_contours{k}))
        fh=drawedgelist({cls_contours{k}}, size(img), LWIDTH, contour_colors{k}, figure_id); axis off; % 'k'
        primitive_names{end+1}=sprintf('c_%g',k);
        ph(end+1)=fh;
        % add arrow:
        arrow_pt = cls_contours{k}(1,:);        fh=plot(arrow_pt(2),arrow_pt(1),'s','LineWidth',LWIDTH+1,'Color',contour_colors{k},'MarkerFaceColor',contour_colors{k});
    end
end

% Regions
for k=1:length(cls_regions)
    if(~isempty(cls_regions{k}))
        region = cls_regions{k};
        top_row = region(1); bottom_row = region(2)+1; left_col = region(3)+2; right_col = region(4);
        rectangle('Position',[left_col,top_row, right_col - left_col, bottom_row - top_row],'EdgeColor',regions_colors{k},'LineWidth',LWIDTH);
        fh=plot(0,0,'LineWidth',LWIDTH,'Color',regions_colors{k});
        primitive_names{end+1}=sprintf('r_%g',k);
        ph(end+1)=fh;
        
    end
end

% Points
for k=1:length(cls_pts)
    if(~isempty(cls_pts{k}))
        pt = cls_pts{k};
        fh=plot(pt(2), pt(1),'*','Color',pts_colors{k},'LineWidth',5);
        primitive_names{end+1}=sprintf('p_%g',k);
        ph(end+1)=fh;
    end
end

hold off;
end


% -------------------------- AUX ------------------------------

% DRAWEDGELIST - plots pixels in edgelists
%
% Usage:    h =  drawedgelist(edgelist, rowscols, lw, col, figno)
%
% Arguments:
%    edgelist   - Cell array of edgelists in the form
%                     { [r1 c1   [r1 c1   etc }
%                        ...
%                        rN cN]   ....]
%    rowscols -   Optional 2 element vector [rows cols] specifying the size
%                 of the image from which edges were detected (used to set
%                 size of plotted image).  If omitted or specified as [] this
%                 defaults to the bounds of the linesegment points
%    lw         - Optional line width specification. If omitted or specified
%                 as [] it defaults to a value of 1;
%    col        - Optional colour specification. Eg [0 0 1] for blue.  This
%                 can also be specified as the string 'rand' to generate a
%                 random color coding for each edgelist so that it is easier
%                 to see how the edges have been broken up into separate
%                 lists. If omitted or specified as [] it defaults to blue
%    figno      - Optional figure number in which to display image.
%
% Returns:
%       h       - Array of handles to each plotted edgelist
%
% See also: EDGELINK, LINESEG

% Copyright (c) 2003-2011 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% February  2003 - Original version
% September 2004 - Revised and updated
% December  2006 - Colour and linewidth specification updated
% January   2011 - Axis setting corrected (thanks to Stzpz)

function h = drawedgelist(edgelist, rowscols, lw, col, figno)
    
    if nargin < 2, rowscols = [1 1]; end
    if nargin < 3, lw = 1; end
    if nargin < 4, col = [0 0 1]; end
    if nargin == 5, figure(figno);  end
    if isempty(rowscols), rowscols = [1 1]; end
    if isempty(lw), lw = 1; end
    if isempty(col), col = [0 0 1]; end    
    
    debug = 0;
    Nedge = length(edgelist);
    h = zeros(length(edgelist),1);
    
    if strcmp(col,'rand')
	colourmp = hsv(Nedge);    % HSV colour map with Nedge entries
	colourmp = colourmp(randperm(Nedge),:);  % Random permutation
	for I = 1:Nedge
	    h(I) = line(edgelist{I}(:,2), edgelist{I}(:,1),...
		 'LineWidth', lw, 'Color', colourmp(I,:));
	end	
    else
	for I = 1:Nedge
	    h(I) = line(edgelist{I}(:,2), edgelist{I}(:,1),...
		 'LineWidth', lw, 'Color', col);
	end	
    end

    if debug
	for I = 1:Nedge
	    mid = fix(length(edgelist{I})/2);
	    text(edgelist{I}(mid,2), edgelist{I}(mid,1),sprintf('%d',I))
	end
    end
    
    % Check whether we need to expand bounds
    minx = 1; miny = 1;
    maxx = rowscols(2); maxy = rowscols(1);

    for I = 1:Nedge
	minx = min(min(edgelist{I}(:,2)),minx);
	miny = min(min(edgelist{I}(:,1)),miny);
	maxx = max(max(edgelist{I}(:,2)),maxx);
	maxy = max(max(edgelist{I}(:,1)),maxy);	
    end	    

    axis('equal'); axis('ij');
    axis([minx maxx miny maxy]);
    
    if nargout == 0
        clear h
    end
    
end
    
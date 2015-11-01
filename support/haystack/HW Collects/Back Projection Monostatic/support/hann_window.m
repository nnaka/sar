function [w] = hann_window(N)
% MIT IAP 2013 Needle in a Haystack Course
% (c) 2013 Massachusetts Institute of Technology

% create a hann (cosine squared) window
w = .5 + .5*cos(2*pi*((1:N).'/(N+1) - .5));
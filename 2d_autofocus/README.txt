README.txt
by Colin Watts and Ian Fletcher


INTRODUCTION:

This is just a brief description of each of the files that can be found in this 
folder.

The overall focus of the information in this folder is focusing. It uses 2D image 
generation code to produce images and also inclues an implomentation of an entropy
minimizing autofocus algorithm for 2D images. This is the basis for all of the 3D 
image generation and autofocus code. A few articles on autofocus algorithms can be
found in this folder as well.


FILE DESCRIPTIONS:

EvtropyVsVariance.m : This file was originally designed to plot entropy of an image
	as the variance of the position error increases. As the autoFocus code
	changed more and more it became more of a way to generate radar data, control
	variance and generate images with pulse sets as well. The auto focus was then
	run seperately. Recommendations are to modify this code to make it do what it
	was originally meant to do (remove the call to autoFocus() entirely) and then
	set up a new file to do what it became.

PulsOnTest.m : This file generates artificial radar returns. Some things to note. 
	Standard error parameters are set such that at sigma = 1 they match the
	typical error distriburion found with GPS testing. The number of pulses
	taken in the aperture can be edited here. 

format_yOut.m : formats the returns from PulsOnTest() so that they are in the same 
	format as what the actual radar would use. 

monoSARwFocus.m : Generates an image from radar returns. Currently set up to work 
	only with artificial data but could easily be edited to use actual data as 
	as well. Creates a set of all pulses taken by the Radar for use in autofocus

autoFocus.m : An implomentation of minimum entropy Autofocus. This is really the 
	focus of this folder. This uses ideas taken from a few of the articles that
	can be found in this folder. Much more detailed descriptions can be found
	in the file itself but on a general sense it will focus a set of radar pulse
	returns as best it can.

PGA.m : This is an unfinished file. This is meant to be an implomentation of a phase
	gradiant autofocus technique but has yet to work. This would ideally take
	much less memory and less space than the entropy minimizing autofocus but
	has some serious issues for now. (version is out of date, TODO: Replace file
	with newest version) 

hann_window.m and fft_interp.m : These are both helper functions. They are both 
	fairly steightforward and don't require much introduction


ARTICLES: 

If you wish to learn more about autofocus algorithms several papers on each have been
provided. 

27_Kragh_Pa.pdf and 00303752.pdf 
	These are both papers on entropy minimizing Autofocus and may be very helpful
	in providing insite on how to imploment a coordinate decent. 

msthesis_bates_lib.pdf and Geoscience and Remote Sensing Letters IEEE  As.pdf
	These are both on PGA, the future of PGA in this project is currently 
	uncertain but these could become useful in the future. 
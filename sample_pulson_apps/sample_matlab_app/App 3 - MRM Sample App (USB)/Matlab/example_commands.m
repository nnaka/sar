
%%
% Look at device manager to determine which COM port MRM is on.
% Just do this once when you first connect.

%%
srl = open_com_port(8)

% NOTE: If you accidentally delete the variable srl, you can recover it
% using the MATLAB command:

%%
srl = instrfind

%%
% Run any of these commands to see scans in various ways.
% Examine the code in these functions and their subfunctions to learn how
% to send requests to MRM and process incoming messages.

%%
plot_one_scn(srl)

%%
Nrqst = 100;
plot_multi_scn(srl,Nrqst)

%%
plot_one_double_scn(srl)

%%
Nrqst = 100;
plot_multi_double_scn(srl,100)

%%
Nrqst = 100;
img_multi_scn(srl,100)

%%
Nrqst = 100;
Nfft = 32;
img_scn_fft(srl,Nrqst,Nfft)

%%
% Close and delete the COM port object.
% Just do this once when you are finished.

%%
fclose(srl);
delete(srl)

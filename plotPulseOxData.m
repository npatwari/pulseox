%
% Script name: plotPulseOxData.m
% Copyright 2021 Neal Patwari
%
% Purpose: 
%   1. Load fake data from Figure 1 of Sjoding "Racial bias..." 2020 paper.
%   2. Plot the data as an example of using the data.
%   3. Calculate some probability of error as an example.
%
% Version History:
%   Version 1.0:  Initial Release.  26 Oct 2021.
%
% License: see LICENSE.md

% Our two hypotheses:
% H0: the "normal"
%     Arterial Oxygen Saturation is >= 88.0
% H1: the "abnormal", what we want to be alarmed about
%     Arterial Oxygen Saturation is < 88.0

% Load data: There are two files separated by race.
% I use _w and _b for the white and Black patient data, respectively
data_w = csvread('oxygenation_w.csv',1,0);
data_b = csvread('oxygenation_b.csv',1,0);

% The 1st column is the pulse ox value.
% The 2nd column is the arterial oxygen saturation.  
%   We take the arterial Ox Sat as the "truth" because it is the "gold standard"
%   for monitoring of oxygen saturation in the blood.
% Each row is one patient.  
pulseOx_w = data_w(:,1);
arterOx_w = data_w(:,2);
pulseOx_b = data_b(:,1);
arterOx_b = data_b(:,2);

% Plot the data
figure(1);
clf();
subplot(1,2,1);  % Subplot with 1 row, 2 columns, currently plotting into #1.
plot(pulseOx_w, arterOx_w,'rx');
grid('on')
set(gca,'FontSize',16);
ylim([68,100])  % Have a uniform y limits for both subplots.
xlabel('Pulse Ox Meast (%)')
ylabel('Arterial Ox Saturation (%)')
legend('White', 'FontSize',16)

subplot(1,2,2)  % Subplot with 1 row, 2 columns, currently plotting into #2.
plot(pulseOx_b, arterOx_b,'rx');
grid('on')
set(gca,'FontSize',16);
ylim([68,100])  % Have a uniform y limits for both subplots.
xlabel('Pulse Ox Meast (%)')
ylabel('Arterial Ox Saturation (%)')
legend('Black', 'FontSize',16)


% Our two hypotheses:
% H0: the "normal"
%     Arterial Oxygen Saturation is >= 88.0
% H1: the "abnormal", what we want to be alarmed about
%     Arterial Oxygen Saturation is < 88.0
%
% As an example, let's find the probability that a white patient 
% has arterial oxygen saturation < 88.0

% Here's a matlab way of finding the indices of the arterOx_w vector where its value < 88.0.
% the find() returns a vector of the indices.  
H1_w_indices   = find(arterOx_w < 88.0);
% We want the probability of the arterial ox sat measurement being < 88.0,
% ie., the proportion.
prob_H1_w      = length(H1_w_indices) / length(arterOx_w)

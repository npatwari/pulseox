#! /usr/bin/env python

#
# Script name: plotPulseOxData.py
# Copyright 2021 Neal Patwari
#
# Purpose: 
#   1. Load fake data from Figure 1 of Sjoding "Racial bias..." 2020 paper.
#   2. Build some detectors of hypoxemia (arterial oxygen saturation of <88%)
#      from pulse oximetry reading.
#   3. Calculate and plot the error performance
#
# Version History:
#   Version 1.0:  Initial Release.  11 Oct 2021.
#
# License: see LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# These are commands I always use to format plots to have a larger font size and
# to refresh automatically as they're changed.
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
plt.ion()


# Our two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0

# Load data: There are two files separated by race.
# I use _w and _b for the white and Black patient data, respectively
data_w = np.loadtxt("temp/white_5.csv", delimiter=',', comments='#')
data_b = np.loadtxt("temp/black_5.csv", delimiter=',', comments='#')

# The 1st column is the pulse ox value.
# The 0th column is the arterial oxygen saturation.  
#   We take the arterial Onp.where(sorted_x0 > sorted_x1[i])[0]x Sat as the "truth" because it is the "gold standard"
#   for monitoring of oxygen saturation in the blood.
# Each row is one patient.  
pulseOx_w = data_w[:,2]
arterOx_w = data_w[:,1]
pulseOx_b = data_b[:,2]
arterOx_b = data_b[:,1]

# Plot the data
plt.figure(1)
plt.clf()
plt.subplot(1,2,1)  # Subplot with 1 row, 2 columns, currently plotting into #1.
plt.plot(pulseOx_w, arterOx_w, 'r.', label="White", linewidth=2)
plt.grid('on')
plt.ylim([68,100])  # Have a uniform y limits for both subplots.
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Arterial Ox Saturation (%)', fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)  # Subplot with 1 row, 2 columns, currently plotting into #2.
plt.plot(pulseOx_b, arterOx_b, 'b.', label="Black", linewidth=2)
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.grid('on')
plt.ylim([68,100])  # Have a uniform y limits for both subplots.
plt.legend(fontsize=16)


# Our two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0
#
# As an example, let's find the probability that a white patient 
# has arterial oxygen saturation < 88.0

# Here's a python way of finding the indices of the arterOx_w vector where its value < 88.0.
# the np.where() returns (strangely) a numpy array, length 1, with the first element being a list
# of the indices.  I take care of that by simply requesting the first element of the numpy array.
H1_w_indices   = np.where(arterOx_w < 88.0)[0]
# We want the probability of the arterial ox sat measurement being < 88.0, ie., the proportion:
# I use *1.0 to make sure that the division is floating point.  This is not necessary in Python 3
# but I do this to be more backwards compatible.
prob_H1_w      = len(H1_w_indices)*1.0 / len(arterOx_w)

print('The probability of H1 for white patients in this data set is ' + str(prob_H1_w))


# Let's group the data by pulse ox measurement and plot statistics.
pulseOxValueList = range(65,101)
bins = len(pulseOxValueList)

meanSaO2_w        = np.zeros(bins)
percentilesSaO2_w = [None]*bins
totalMeasts_w     = [None]*bins
stdSaO2_w         = [None]*bins
for (i, pulseOxVal) in enumerate(pulseOxValueList):
    indices = np.where(np.abs(pulseOx_w-pulseOxVal) < 0.1)[0]
    meanSaO2_w[i] = np.mean(arterOx_w[indices])
    percentilesSaO2_w[i] = np.percentile(arterOx_w[indices], [25,50,75])
    totalMeasts_w[i] = len(indices)
    stdSaO2_w[i] = np.std(arterOx_w[indices],ddof=1)


medianSaO2_w = [p[1] for p in percentilesSaO2_w]
per25SaO2_w = [p[0] for p in percentilesSaO2_w]
per75SaO2_w = [p[2] for p in percentilesSaO2_w]
plt.figure(2)
plt.clf()
plt.plot(pulseOxValueList, per75SaO2_w, 'gv', label="75th %ile SaO2 W", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_w, 'kx', label="Median SaO2 W", linewidth=2)
plt.plot(pulseOxValueList, meanSaO2_w, 'b-', label="Mean SaO2 W", linewidth=1)
plt.plot(pulseOxValueList, per25SaO2_w, 'r^', label="25th %ile SaO2 W", linewidth=2)
plt.grid('on')
plt.xlim([80,100])  # Have a uniform x limits for both subplots.
plt.ylim([70,100])  # Have a uniform y limits for both subplots.
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Arterial O2 Sat (%)', fontsize=16)
plt.legend(fontsize=14)

plt.figure(3)
plt.clf()
plt.semilogy(pulseOxValueList, totalMeasts_w, 'kx', label="Measurements W", linewidth=2)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Number of Measts', fontsize=16)
plt.legend(fontsize=16)

meanSaO2_b        = np.zeros(bins)
percentilesSaO2_b = [None]*bins
totalMeasts_b     = [None]*bins
stdSaO2_b         = [None]*bins
for (i, pulseOxVal) in enumerate(pulseOxValueList):
    indices = np.where(np.abs(pulseOx_b-pulseOxVal) < 0.1)[0]
    if len(indices) > 0:
        meanSaO2_b[i] = np.mean(arterOx_b[indices])
        totalMeasts_b[i] = len(indices)
    if len(indices) > 3:
        percentilesSaO2_b[i] = np.percentile(arterOx_b[indices], [25,50,75])
        stdSaO2_b[i] = np.std(arterOx_b[indices],ddof=1)
    else:
        percentilesSaO2_b[i] = [None, None, None]

medianSaO2_b = [p[1] for p in percentilesSaO2_b]
per25SaO2_b = [p[0] for p in percentilesSaO2_b]
per75SaO2_b = [p[2] for p in percentilesSaO2_b]
plt.figure(4)
plt.clf()
plt.plot(pulseOxValueList, per75SaO2_b, 'gv', label="75th %ile SaO2 B", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_b, 'kx', label="Median SaO2 B", linewidth=2)
plt.plot(pulseOxValueList, meanSaO2_b, 'b-', label="Mean SaO2 B", linewidth=1)
plt.plot(pulseOxValueList, per25SaO2_b, 'r^', label="25th %ile SaO2 B", linewidth=2)
plt.grid('on')
plt.xlim([80,100])  # Have a uniform x limits for both subplots.
plt.ylim([70,100])  # Have a uniform y limits for both subplots.
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Arterial O2 Sat (%)', fontsize=16)
plt.legend(fontsize=16)

plt.figure(5)
plt.clf()
plt.semilogy(pulseOxValueList, totalMeasts_b, 'kx', label="Measurements B", linewidth=2)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Number of Measts', fontsize=16)
plt.legend(fontsize=14)

plt.figure(6)
plt.clf()
plt.plot(pulseOxValueList, stdSaO2_w, 'rx', label="Stdev W", linewidth=2)
plt.plot(pulseOxValueList, stdSaO2_b, 'kx', label="Stdev B", linewidth=2)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Std dev of SaO2 Measts', fontsize=16)
plt.xlim([80,100])  # Have a uniform x limits for both subplots.
plt.legend(fontsize=14)


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
data_w = np.loadtxt("oxygenation_w.csv", delimiter=', ', comments='#')
data_b = np.loadtxt("oxygenation_b.csv", delimiter=', ', comments='#')

# The 0th column is the pulse ox value.
# The 1st column is the arterial oxygen saturation.  
#   We take the arterial Ox Sat as the "truth" because it is the "gold standard"
#   for monitoring of oxygen saturation in the blood.
# Each row is one patient.  
pulseOx_w = data_w[:,0]
arterOx_w = data_w[:,1]
pulseOx_b = data_b[:,0]
arterOx_b = data_b[:,1]

# Plot the data
plt.figure(1)
plt.clf()
plt.subplot(1,2,1)  # Subplot with 1 row, 2 columns, currently plotting into #1.
plt.plot(pulseOx_w, arterOx_w, 'rx', label="White", linewidth=2)
plt.grid('on')
plt.ylim([68,100])  # Have a uniform y limits for both subplots.
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Arterial Ox Saturation (%)', fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)  # Subplot with 1 row, 2 columns, currently plotting into #2.
plt.plot(pulseOx_b, arterOx_b, 'bx', label="Black", linewidth=2)
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

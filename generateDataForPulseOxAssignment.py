#! /usr/bin/env python

#
# Script name: generateDataForPulseOxAssignment.py
# Copyright 2021 Neal Patwari
#
# Purpose: Generate fake data from Figure 1 of 
#   https://www.nejm.org/doi/full/10.1056/NEJMc2029240
#   Sjoding, Michael W., Robert P. Dickson, Theodore J. Iwashyna, 
#   Steven E. Gay, and Thomas S. Valley.  "Racial bias in pulse oximetry 
#   measurement". New England Journal of Medicine 383, no. 25 
#   (2020): 2477-2478.
#
# Version History:
#   Version 1.0:  Initial Release.  11 Oct 2021.
#
# License: see LICENSE.md


from metalogistic.main import MetaLogistic
import matplotlib.pyplot as plt
import numpy as np
from random import random

bins = 8

# The box plot provides marks at 25%ile, median, and 75%ile.
ps = [.25, .5, .75]

# These are just my best guess reading from Figure 1 of
# https://www.nejm.org/doi/full/10.1056/NEJMc2029240
# Figure 1. Accuracy of Pulse Oximetry in Measuring Arterial Oxygen Saturation, According to Race.
xs_white_list = [
    [86.2, 88.7, 91.7],  # Pulseox = 89
    [87.0, 89.0, 91.5],  # Pulseox = 90
    [88.5, 90.8, 92.5],  # Pulseox = 91
    [89.6, 91.6, 93.7],  # Pulseox = 92
    [90.2, 92.3, 93.9],  # Pulseox = 93
    [91.0, 92.5, 94.4],  # Pulseox = 94
    [92.3, 93.6, 95.0],  # Pulseox = 95
    [93.1, 94.3, 95.5]] # Pulseox = 96

xs_Black_list = [
    [84.5, 86.3, 87.5],  # Pulseox = 89
    [85.5, 86.8, 90.5],  # Pulseox = 90
    [85.5, 87.5, 90.2],  # Pulseox = 91
    [87.2, 89.1, 91.0],  # Pulseox = 92
    [88.6, 90.3, 92.0],  # Pulseox = 93
    [89.2, 91.2, 92.4],  # Pulseox = 94
    [90.1, 91.8, 92.9],  # Pulseox = 95
    [90.1, 92.3, 94.2]] # Pulseox = 96

pulseOxRange = range(89,97)
NoOfPairedMeasts_white = [92, 178, 231, 314, 438, 556, 653, 817]
NoOfPairedMeasts_Black = [20, 52, 59, 83, 127, 126, 188, 225]



# Init the data lists
data_w = []
data_b = []

# For each possible pulseox value...
for i in range(bins):

	# Generate the CDF of the SpO2 conditioned on pulseox value.
	# This is from the box & whisker plot in Figure 1
	# There is one box for white patients, one for Black patients.
	# The value is a percentage of saturation, so must be in [0, 100]
	m_white = MetaLogistic(ps, xs_white_list[i], lbound=0, ubound=100)

	# Generate a random SpO2 from the calculated CDF by generating a 
	# uniform random value, and plugging that into the inverse CDF function.
	# Do this for as many values as were recorded in the study 
	for j in range(NoOfPairedMeasts_white[i]):
		data_w.append([pulseOxRange[i], m_white.quantile(random())])

    # Repeat for Black patient data
	m_Black = MetaLogistic(ps, xs_Black_list[i], lbound=0, ubound=100)
	for j in range(NoOfPairedMeasts_Black[i]):
		data_b.append([pulseOxRange[i], m_Black.quantile(random())])

# Save the output data
np.savetxt("oxygenation_w.csv", data_w, delimiter=', ', fmt='%5.2f',header="PulseOx, Arterial oxygen saturation")
np.savetxt("oxygenation_b.csv", data_b, delimiter=', ', fmt='%5.2f',header="PulseOx, Arterial oxygen saturation")

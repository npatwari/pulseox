
#
# Script name: pulseOxAssignment.py
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
from matplotlib.ticker import MultipleLocator


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
plt.ion()

#
# Input two lists of data 
#   x_0: X values given hypothesis H0 is true
#   x_1: X values given hypothesis H1 is true
#   plotOption: True to have this function plot the ROC curve
#
# Output: a plot of the probability of false alarm (P_FA) on the x-axis
#   vs. the probability of correct detection (P_D) on the y-axis.  This is
#   called the "receiver operating characteristic" or ROC curve.
#
# Returns: P_FA, P_D numpy arrays which can be plotted.
#
def computeROC(x_0, x_1, plotOption=True):
	# Convert inputs to vectors (in case matrices are input)
	input0    = np.array(x_0)
	input1    = np.array(x_1)

	# This code assumes that the mean of x given H0 is LESS THAN
	# the mean of x given H1.  If that's not true, then multiply 
	# everything by -1 so that it is true.
	if np.mean(input0) > np.mean(input1):
		input0 = -1.0*input0
		input1 = -1.0*input1

	len0      = len(input0)
	len1      = len(input1)
	if len0==0 or len1==0:
		print("Length of both inputs to computeROC must be positive")

	# Sorting produces the x-values in the complementary
	# cumulative distribution function (CCDF)
	sorted_x0 = np.sort(input0)
	sorted_x1 = np.sort(input1)

	# Calculate the y-values in the CCDF.  Rather than going
	# from 1 down to 0, we count down from 1-ep/2 to ep/2 
	# to account for the fact that we never really have a 
	# probability of detection or false alarm of 0 or 1 with
	# a finite quantity of data points.
	ep0       = 1.0/len0
	ep1       = 1.0/len1
	ccdf0     = np.arange( 1-ep0/2, 0, -ep0)
	ccdf1     = np.arange( 1-ep1/2, 0, -ep1)

	# We need a probability of false alarm (p_FA) vector that
	# is aligned with the CCDF for H_1 (ccdf1).  p_FA(i) answers
	# the question, what is the probability of false alarm at
	# the same threshold that produces a probability of detection 
	# of ccdf1(i)?  The value sorted_x1(i) is the threshold for 
	# which results in ccdf1(i) probability of detection. 
	# This same threshold would result in a p_FA(i) value 
	# equal to the proportion of values in H_0 > it.
	p_FA = np.zeros(len1)
	for i in range(len1):
		temp = np.where(sorted_x0 > sorted_x1[i])[0]
		if len(temp)==0:
			p_FA[i] = ccdf0[-1]
		else:
			p_FA[i] = ccdf0[temp[0]]

	# Plot the ROC curve: P[False Alarm] vs. P[Correct Detection]
	if plotOption:
		plt.plot(p_FA, ccdf1, 'r-o', linewidth=2)
		plt.grid('on')
		#plt.xlim(100, 2000)
		plt.xlabel('Probability of False Alarm', fontsize=16)
		plt.ylabel('Probability of Correct Detection', fontsize=16)
		plt.show()

	return p_FA, ccdf1


# main

# Any detection problem starts with two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0

# Load data
data_w = np.loadtxt("./Data 4-22/white_10.csv", delimiter=',', comments='%')
data_b = np.loadtxt("./Data 4-22/black_10.csv", delimiter=',', comments='%')
data_a = np.loadtxt("./Data 4-22/asian_10.csv", delimiter=',', comments='%')
data_h = np.loadtxt("./Data 4-22/hispanic_10.csv", delimiter=',', comments='%')
data_n = np.loadtxt("./Data 4-22/native_american_10.csv", delimiter=',', comments='%')

# The 1st column is the pulse ox value.
# The 0th column is the arterial oxygen saturation.  
#   We take the arterial Onp.where(sorted_x0 > sorted_x1[i])[0]x Sat as the "truth" because it is the "gold standard"
#   for monitoring of oxygen saturation in the blood.
# Each row is one patient.  
pulseOx_w = data_w[:,2]
arterOx_w = data_w[:,1]
pulseOx_b = data_b[:,2]
arterOx_b = data_b[:,1]
pulseOx_a = data_a[:,2]
arterOx_a = data_a[:,1]
pulseOx_h = data_h[:,2]
arterOx_h = data_h[:,1]
pulseOx_n = data_n[:,2]
arterOx_n = data_n[:,1]
pulseOx_all = np.concatenate((pulseOx_w, pulseOx_b, pulseOx_a, pulseOx_h, pulseOx_n))
arterOx_all = np.concatenate((arterOx_w, arterOx_b, arterOx_a, arterOx_h, arterOx_n))

# Find all pulseOx values for which H0 is genuinely true, ie, arterOx >= 88
temp      = np.where(arterOx_b >= 88.0)[0]
pulseOx_b_H0 = pulseOx_b[temp]
temp      = np.where(arterOx_a >= 88.0)[0]
pulseOx_a_H0 = pulseOx_a[temp]
temp      = np.where(arterOx_w >= 88.0)[0]
pulseOx_w_H0 = pulseOx_w[temp]
temp      = np.where(arterOx_all >= 88.0)[0]
pulseOx_all_H0 = pulseOx_all[temp]

# Find all pulseOx values for which H1 is genuinely true, ie, arterOx < 88
temp      = np.where(arterOx_b < 88.0)[0]
pulseOx_b_H1 = pulseOx_b[temp]
temp      = np.where(arterOx_a < 88.0)[0]
pulseOx_a_H1 = pulseOx_a[temp]
temp      = np.where(arterOx_w < 88.0)[0]
pulseOx_w_H1 = pulseOx_w[temp]
temp      = np.where(arterOx_all < 88.0)[0]
pulseOx_all_H1 = pulseOx_all[temp]

# What is the probability of hypoxemia?
print("P[hypoxemia | Black] = " + str( len(pulseOx_b_H1) *1.0 / len(pulseOx_b)))
print("P[hypoxemia | Asian] = " + str( len(pulseOx_a_H1) *1.0 / len(pulseOx_a)))
print("P[hypoxemia | White] = " + str( len(pulseOx_w_H1) *1.0 / len(pulseOx_w)))
print("P[hypoxemia] = " + str( len(pulseOx_all_H1) *1.0 / len(pulseOx_all)))

# 1. What if a single pulseOx threshold must be used for all patients?
#
# There are only integer pulseox values, so use 88.5, 89.5, ..., 96.5 
# as possible thresholds.  Initialize the vectors (for false alarm and detection,
# for white and Black patients).
threshold_list = np.concatenate((np.array([60.5, 70.5, 76.5, 80.5, 82.5]), np.arange(84.5, 99.5, 1.0)))
p_FA_b    = np.zeros(len(threshold_list))
p_FA_a    = np.zeros(len(threshold_list))
p_FA_w    = np.zeros(len(threshold_list))
p_FA_all  = np.zeros(len(threshold_list))
p_D_b     = np.zeros(len(threshold_list))
p_D_a     = np.zeros(len(threshold_list))
p_D_w     = np.zeros(len(threshold_list))
p_D_all   = np.zeros(len(threshold_list))
for i, threshold in enumerate(threshold_list):

    # For each threshold, calculate the probability of false alarm, ie., 
    # the probability of raising the alarm when H0 is true 
	p_FA_b[i] = float(np.count_nonzero(pulseOx_b_H0 < threshold)) / len(pulseOx_b_H0)
	p_FA_a[i] = float(np.count_nonzero(pulseOx_a_H0 < threshold)) / len(pulseOx_a_H0)
	p_FA_w[i] = float(np.count_nonzero(pulseOx_w_H0 < threshold)) / len(pulseOx_w_H0)
	p_FA_all[i] = float(np.count_nonzero(pulseOx_all_H0 < threshold)) / len(pulseOx_all_H0)
    # For each threshold, calculate the probability of correct detection, ie., 
    # the probability of raising the alarm when H1 is true 
	p_D_b[i]  = float(np.count_nonzero(pulseOx_b_H1 < threshold)) / len(pulseOx_b_H1)
	p_D_a[i]  = float(np.count_nonzero(pulseOx_a_H1 < threshold)) / len(pulseOx_a_H1)
	p_D_w[i]  = float(np.count_nonzero(pulseOx_w_H1 < threshold)) / len(pulseOx_w_H1)
	p_D_all[i] = float(np.count_nonzero(pulseOx_all_H1 < threshold)) / len(pulseOx_all_H1)

# Sum of FA probability and MD probability
p_TOT_b = p_FA_b - p_D_b + 1.0 
p_TOT_w = p_FA_w - p_D_w + 1.0 
p_TOT_all = p_FA_all - p_D_all + 1.0 

print("Optimal threshold for Black patients: {:.1f}".format(threshold_list[np.argmin(p_TOT_b)]))
print("Optimal threshold for White patients: {:.1f}".format(threshold_list[np.argmin(p_TOT_w)]))
print("Optimal threshold for All patients: {:.1f}".format(threshold_list[np.argmin(p_TOT_all)]))

print("Threshold: " + str(threshold_list[3]))
print("       | P[false alarm] | P[correct detection] ")
print("Black  | " + "{:1.4f}".format(p_FA_b[3]) + " | " + "{:1.4f}".format(p_D_b[3]))
print("White  | " + "{:1.4f}".format(p_FA_w[3]) + " | " + "{:1.4f}".format(p_D_w[3]))
print("All    | " + "{:1.4f}".format(p_FA_all[3]) + " | " + "{:1.4f}".format(p_D_all[3]))

# Plot the results
plt.figure(1)
for i, threshold in enumerate(threshold_list):
	# Put the threshold on each dot, connect the white/Black points for 
	# that correspond to the same threshold.
	if threshold < 95:
		plt.text(p_FA_w[i]-0.0035, p_D_w[i], str(threshold), horizontalalignment='right')
	if threshold < 96:
		plt.text(p_FA_a[i]+0.004, p_D_a[i]-0.015, str(threshold), horizontalalignment='left')
	plt.plot([p_FA_a[i],p_FA_w[i]], [p_D_a[i],p_D_w[i]], 'b-', linewidth=2)

plt.plot(p_FA_w, p_D_w, marker="D", markeredgecolor='k', \
    markerfacecolor='red', label="White", linewidth=0)
plt.plot(p_FA_a, p_D_a, 'ko', marker="X", markeredgecolor='k', \
    markerfacecolor='blue', label="Asian", linewidth=0)
plt.grid('on', which='both')
# put in minor grid lines
for x in np.arange(0.02, 0.70, 0.04):
	plt.plot([x, x], [-0.01, 1], 'k',linewidth=0.25)
for y in np.arange(0.05, 1.0, 0.1):
	plt.plot([-0.13,1], [y, y], 'k',linewidth=0.25)
plt.xlabel('Prob. of False Alarm / Type I Error', fontsize=16)
plt.ylabel('Prob. of Detection / True Positive', fontsize=16)
plt.xticks(np.arange(0, 0.71, 0.04)) 
plt.yticks(np.arange(0, 0.71, 0.1))
plt.legend(fontsize=14,loc="lower right")

plt.xlim([-0.013, 0.21])
plt.ylim([-0.01, 0.71])



# 2. What if we just plot, separately, the ROC curves for Black & white patients?
# #
# # Call computeROC once for the Black patient data, once for the white patient data
# p_FA_b, P_D_b = computeROC( pulseOx_b_H0, pulseOx_b_H1, plotOption=False)
# p_FA_w, P_D_w = computeROC( pulseOx_w_H0, pulseOx_w_H1, plotOption=False)
# xmax = max(np.max(p_FA_b), np.max(p_FA_w)) 

# # Plot the two ROC curves.
# plt.figure(2)
# plt.plot(p_FA_w, P_D_w, 'rs-', label="White", linewidth=2)
# plt.plot(p_FA_b, P_D_b, 'ko-', label="Black", linewidth=2)
# plt.grid('on')
# plt.xlabel('Probability of False Alarm', fontsize=16)
# plt.ylabel('Probability of Correct Detection', fontsize=16)
# plt.xticks(np.arange(0, xmax+0.09999, 0.1))
# plt.yticks(np.arange(0, 1.01, 0.1))
# plt.legend(fontsize=16)
# plt.show()


# 3. But what if we add a correction factor to each pulse ox measurement, 
#    where the additive factor is based on the race of the patient?
#    Surely that would help?  Tl:dr; no that doesn't help.

# Copying in the 25th, median, and 75th percentiles from Figure 1.
# xs_white_list = [
#     [86.2, 88.7, 91.7],  # Pulseox = 89
#     [87.0, 89.0, 91.5],  # Pulseox = 90
#     [88.5, 90.8, 92.5],  # Pulseox = 91
#     [89.6, 91.6, 93.7],  # Pulseox = 92
#     [90.2, 92.3, 93.9],  # Pulseox = 93
#     [91.0, 92.5, 94.4],  # Pulseox = 94
#     [92.3, 93.6, 95.0],  # Pulseox = 95
#     [93.1, 94.3, 95.5]] # Pulseox = 96

# xs_Black_list = [
#     [84.5, 86.3, 87.5],  # Pulseox = 89
#     [85.5, 86.8, 90.5],  # Pulseox = 90
#     [85.5, 87.5, 90.2],  # Pulseox = 91
#     [87.2, 89.1, 91.0],  # Pulseox = 92
#     [88.6, 90.3, 92.0],  # Pulseox = 93
#     [89.2, 91.2, 92.4],  # Pulseox = 94
#     [90.1, 91.8, 92.9],  # Pulseox = 95
#     [90.1, 92.3, 94.2]] # Pulseox = 96

# # Create a corrected pulseOx vector, for Black and white
# # the correction is the statistical bias between pulseox value and 
# # the median arterial oxygen saturation.
# pulseOxRange = range(89,97)
# bias_w = np.zeros(len(pulseOxRange))
# bias_b = np.zeros(len(pulseOxRange))
# pulseOx_w_corr = np.copy(pulseOx_w)
# pulseOx_b_corr = np.copy(pulseOx_b)
# for i in range(len(pulseOxRange)):
#     # Subtract the bias for white patient data
# 	bias_w[i] = pulseOxRange[i] - xs_white_list[i][1]
# 	temp = np.where(np.abs(pulseOx_w - pulseOxRange[i]) < 0.5)[0]
# 	pulseOx_w_corr[temp] -= bias_w[i]
#     # Subtract the bias for black patient data
# 	bias_b[i] = pulseOxRange[i] - xs_Black_list[i][1]
# 	temp = np.where(np.abs(pulseOx_b - pulseOxRange[i]) < 0.5)[0]
# 	pulseOx_b_corr[temp] -= bias_b[i]


# # Find all pulseOx values for which H0 is genuinely true, ie, arterOx >= 88
# temp = np.where(arterOx_b >= 88.0)[0]
# pulseOx_b_corr_H0 = pulseOx_b_corr[temp]
# temp = np.where(arterOx_w >= 88.0)[0]
# pulseOx_w_corr_H0 = pulseOx_w_corr[temp]

# # Find all pulseOx values for which H1 is genuinely true, ie, arterOx < 88
# temp = np.where(arterOx_b < 88.0)[0]
# pulseOx_b_corr_H1 = pulseOx_b_corr[temp]
# temp = np.where(arterOx_w < 88.0)[0]
# pulseOx_w_corr_H1 = pulseOx_w_corr[temp]

# # Call computeROC once for the Black patient data, once for the white patient data
# p_FA_b, P_D_b = computeROC( pulseOx_b_corr_H0, pulseOx_b_corr_H1, plotOption=False)
# p_FA_w, P_D_w = computeROC( pulseOx_w_corr_H0, pulseOx_w_corr_H1, plotOption=False)
# xmax = max(np.max(p_FA_b), np.max(p_FA_w)) 

# plt.figure(3)
# plt.plot(p_FA_w, P_D_w, 'ms-', label="White Corr", linewidth=2)
# plt.plot(p_FA_b, P_D_b, 'bo-', label="Black Corr", linewidth=2)
# plt.grid('on')
# plt.xlabel('Probability of False Alarm', fontsize=16)
# plt.ylabel('Probability of Correct Detection', fontsize=16)
# plt.xticks(np.arange(0, xmax+0.09999, 0.1))
# plt.yticks(np.arange(0, 1.01, 0.1))
# plt.legend(fontsize=16)
# plt.show()


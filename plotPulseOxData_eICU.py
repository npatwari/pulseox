#! /usr/bin/env python

#
# Script name: plotPulseOxData.py
# Copyright 2021 Neal Patwari
#
# Purpose: 
#   1. Load real data from eICU database, part of Sjoding "Racial bias..." 2020 paper.
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
import statistics
import matplotlib
from scipy.stats import median_abs_deviation
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import norm

# These are commands I always use to format plots to have a larger font size and
# to refresh automatically as they're changed.
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
plt.ion()


# return pvalue for a test of difference in binomial probabilities using the normal
# approximation.  
# I assume tp1 < p2
def binomialTest(p1, p2, n1, n2, onesided=True):

    phat = (n2*p2 + n1*p1)/ (n2 + n1)
    zvalue = (p2 - p1) / np.sqrt(phat*(1 - phat)*((1/n1) + (1/n2)))

    if onesided==True:
        # the "-zvalue" is to get the complementary cdf of the standard normal
        pvalue = norm.cdf(-zvalue)
    else:
        # the "-np.abs(zvalue)" is to get the complementary cdf of the standard normal
        #   that works for both negative and positive tails.
        # the 2.0* is to give the p value for a two-sided test.
        pvalue = 2.0*norm.cdf(-np.abs(zvalue))

    return pvalue


# def computePMF(bincenters, values):
#     bins = len(bincenters)
#     dx   = bincenters[1]-bincenters[0]
#     binstarts = bincenters - (dx/2.0)
#     binends   = bincenters + (dx/2.0)
#     pmf       = np.zeros(bins)
#     for i in range(bins):



# From https://stackoverflow.com/questions/59747313/how-to-plot-confidence-interval-in-python
def plot_confidence_interval(x, values, z=1.96, marker='o', color='#2187bb', horizontal_line_width=0.5, legendlabel=''):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / np.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    #marker_style = dict(color='tab:blue', linestyle=':', marker='o', markersize=15, markerfacecoloralt='tab:red')
    ax = plt.plot(x, mean, marker, color='black', markerfacecolor=color, label=legendlabel)

    return ax, mean #, [top, bottom]


def groupBySPO2(raceStr, pulseOxValueList, pulseOx, arterOx):

    # Let's group the data by pulse ox measurement and plot statistics.
    pulseOxValueList = range(65,101)
    bins = len(pulseOxValueList)

    allSaO2      = [np.array([])]*bins
    biasSaO2     = [None]*bins
    meanSaO2     = np.zeros(bins)
    medianSaO2   = [None]*bins
    per25SaO2    = [None]*bins
    per75SaO2    = [None]*bins
    totalMeasts  = [None]*bins
    stdSaO2      = [None]*bins
    for (i, pulseOxVal) in enumerate(pulseOxValueList):
        indices = np.where(np.abs(pulseOx-pulseOxVal) < 0.1)[0]
        if len(indices)>0:
            allSaO2[i]     = arterOx[indices]
            biasSaO2[i]    = pulseOxVal - arterOx[indices]
        if len(indices)>3:
            meanSaO2[i]    = np.mean(arterOx[indices])
            temp           = np.percentile(arterOx[indices], [25,50,75])
            medianSaO2[i]  = temp[1]
            per25SaO2[i]   = temp[0]
            per75SaO2[i]   = temp[2]
            totalMeasts[i] = len(indices)
            stdSaO2[i]     = np.std(arterOx[indices],ddof=1)

    # plt.figure()
    # plt.clf()
    # plt.plot(pulseOxValueList, per75SaO2, 'gv', label="75th %ile SaO2", linewidth=2)
    # plt.plot(pulseOxValueList, medianSaO2, 'kx', label="Median SaO2", linewidth=2)
    # plt.plot(pulseOxValueList, meanSaO2, 'b-', label="Mean SaO2", linewidth=1)
    # plt.plot(pulseOxValueList, per25SaO2, 'r^', label="25th %ile SaO2", linewidth=2)
    # plt.title(raceStr)
    # plt.grid('on')
    # plt.xlim([80,100])  # Have a uniform x limits for both subplots.
    # plt.ylim([70,100])  # Have a uniform y limits for both subplots.
    # plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
    # plt.ylabel('Arterial O2 Sat (%)', fontsize=16)
    # plt.legend(fontsize=14)

    # plt.figure()
    # plt.clf()
    # plt.semilogy(pulseOxValueList, totalMeasts, 'kx', label=raceStr, linewidth=2)
    # plt.grid('on')
    # plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
    # plt.ylabel('Number of Measts', fontsize=16)
    # plt.legend(fontsize=16)

    return meanSaO2, per25SaO2, medianSaO2, per75SaO2, totalMeasts, stdSaO2, allSaO2, biasSaO2


# Our two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0

# Load data: There are two files separated by race.
# I use _w and _b for the white and Black patient data, respectively
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

# Create a table of data (measurements, percentage of the whole set) by race/ethnicity.
len_w = len(pulseOx_w)
len_b = len(pulseOx_b)
len_a = len(pulseOx_a)
len_h = len(pulseOx_h)
len_n = len(pulseOx_n)
total = len_w + len_b + len_a+ len_h + len_n
print("Data pairs total:")
print("White: " + str(len_w) + ", Percent: " + str(len_w*100.0/total) + "%")
print("Black: " + str(len_b) + ", Percent: " + str(len_b*100.0/total) + "%")
print("Asian: " + str(len_a) + ", Percent: " + str(len_a*100.0/total) + "%")
print("Hispanic: " + str(len_h) + ", Percent: " + str(len_h*100.0/total) + "%")
print("Native Am.: " + str(len_n) + ", Percent: " + str(len_n*100.0/total) + "%")

# Perform two-sided T-tests to compare each race/ethnicity other than white, to white 
# Null hypothesis: identical means
# H1: different means.
# Do not assume identical variances.
errors_w = pulseOx_w - arterOx_w
errors_b = pulseOx_b - arterOx_b
errors_a = pulseOx_a - arterOx_a
errors_h = pulseOx_h - arterOx_h
errors_n = pulseOx_n - arterOx_n
errors_all = np.concatenate((errors_w, errors_b, errors_a, errors_h, errors_n))
stat_b, pvalue_b = ttest_ind(errors_w, errors_b, equal_var=False, nan_policy="omit", alternative="two-sided")
stat_a, pvalue_a = ttest_ind(errors_w, errors_a, equal_var=False, nan_policy="omit", alternative="two-sided")
stat_h, pvalue_h = ttest_ind(errors_w, errors_h, equal_var=False, nan_policy="omit", alternative="two-sided")
stat_n, pvalue_n = ttest_ind(errors_w, errors_n, equal_var=False, nan_policy="omit", alternative="two-sided")

# Compute a one-way F-test to compare the variance of error in each race/ethnicity other than white, to white 
# Null hypothesis: identical variances
# H1: variance of group is greater than variance of white.
fstat_b, fpvalue_b = f_oneway(errors_w, errors_b)
fstat_a, fpvalue_a = f_oneway(errors_w, errors_a)
fstat_h, fpvalue_h = f_oneway(errors_w, errors_h)
fstat_n, fpvalue_n = f_oneway(errors_w, errors_n)

# Create a table of average SpO2 bias by race/ethnicity.
print("Pulse Ox Bias: Average & Median vs. Race/Ethnicity")
print("White & " + "{:.2f}".format(np.median(errors_w)) + "\\% & " + "{:.2f}".format(np.mean(errors_w)) + "\\% & \\\\")
print("Black & " + "{:.2f}".format(np.median(errors_b)) + "\\% & " + "{:.2f}".format(np.mean(errors_b)) + "\\% & *** \\\\")
print("Asian & " + "{:.2f}".format(np.median(errors_a)) + "\\% & " + "{:.2f}".format(np.mean(errors_a)) + "\\% & *** \\\\")
print("Hispanic & " + "{:.2f}".format(np.median(errors_h)) + "\\% & " + "{:.2f}".format(np.mean(errors_h)) + "\\% & \\\\")
print("Native Am. & " + "{:.2f}".format(np.median(errors_n)) + "\\% & " + "{:.2f}".format(np.mean(errors_n)) + "\\% & \\\\")
print("All Data & " + "{:.2f}".format(np.median(errors_all)) + "\\% & " + "{:.2f}".format(np.mean(errors_all)) + "\\% & \\\\")

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

# separate patient data by race and by SpO2 value
meanSaO2_w, per25SaO2_w, medianSaO2_w, per75SaO2_w, totalMeast_w, stdSaO2_w, allSaO2_w, biasSaO2_w = \
    groupBySPO2('White', pulseOxValueList, pulseOx_w, arterOx_w)

meanSaO2_b, per25SaO2_b, medianSaO2_b, per75SaO2_b, totalMeast_b, stdSaO2_b, allSaO2_b, biasSaO2_b = \
    groupBySPO2('Black', pulseOxValueList, pulseOx_b, arterOx_b)

meanSaO2_a, per25SaO2_a, medianSaO2_a, per75SaO2_a, totalMeast_a, stdSaO2_a, allSaO2_a, biasSaO2_a = \
    groupBySPO2('Asian', pulseOxValueList, pulseOx_a, arterOx_a)

meanSaO2_h, per25SaO2_h, medianSaO2_h, per75SaO2_h, totalMeast_h, stdSaO2_h, allSaO2_h, biasSaO2_h = \
    groupBySPO2('Hispanic', pulseOxValueList, pulseOx_h, arterOx_h)

meanSaO2_n, per25SaO2_n, medianSaO2_n, per75SaO2_n, totalMeast_n, stdSaO2_n, allSaO2_n, biasSaO2_n = \
    groupBySPO2('Native American', pulseOxValueList, pulseOx_n, arterOx_n)


# Compute the difference between Sa02 value and the mean SaO2 given SpO2.
# For each SpO2.
err_w = [np.array([])]*len(pulseOxValueList)
err_b = [np.array([])]*len(pulseOxValueList)
err_a = [np.array([])]*len(pulseOxValueList)
err_h = [np.array([])]*len(pulseOxValueList)
err_n = [np.array([])]*len(pulseOxValueList)
errm_w = [np.array([])]*len(pulseOxValueList)
errm_b = [np.array([])]*len(pulseOxValueList)
errm_a = [np.array([])]*len(pulseOxValueList)
errm_h = [np.array([])]*len(pulseOxValueList)
errm_n = [np.array([])]*len(pulseOxValueList)
for (i,spo2) in enumerate(pulseOxValueList):
    if len(allSaO2_w[i])>1:
        err_w[i] = allSaO2_w[i] - np.mean(allSaO2_w[i])
        errm_w[i] = allSaO2_w[i] - np.median(allSaO2_w[i])
    if len(allSaO2_b[i])>1:
        err_b[i] = allSaO2_b[i] - np.mean(allSaO2_b[i])
        errm_b[i] = allSaO2_b[i] - np.median(allSaO2_b[i])
    if len(allSaO2_a[i])>1:
        err_a[i] = allSaO2_a[i] - np.mean(allSaO2_a[i])
        errm_a[i] = allSaO2_a[i] - np.median(allSaO2_a[i])
    if len(allSaO2_h[i])>1:
        err_h[i] = allSaO2_h[i] - np.mean(allSaO2_h[i])
        errm_h[i] = allSaO2_h[i] - np.median(allSaO2_h[i])
    if len(allSaO2_n[i])>1:
        err_n[i] = allSaO2_n[i] - np.mean(allSaO2_b[i])
        errm_n[i] = allSaO2_n[i] - np.median(allSaO2_b[i])



plt.figure()
plt.clf()
plt.plot(pulseOxValueList, stdSaO2_w, marker="D", markeredgecolor='k', \
    markerfacecolor='red', label="White", linewidth=0)
plt.plot(pulseOxValueList, stdSaO2_h, marker="*", markeredgecolor='k', \
    markerfacecolor='orange', label="Hispanic", linewidth=0)
plt.plot(pulseOxValueList, stdSaO2_n, marker="s", markeredgecolor='k', \
    markerfacecolor='green',  label="Native Am.", linewidth=0)
plt.plot(pulseOxValueList, stdSaO2_a, marker="X", markeredgecolor='k', \
    markerfacecolor='blue', label="Asian", linewidth=0)
plt.plot(pulseOxValueList, stdSaO2_b, marker="o", markeredgecolor='k', \
    markerfacecolor='purple', label="Black", linewidth=0)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Std dev of SaO2 Measts', fontsize=16)
plt.xlim([86.5,100])  # Have a uniform x limits for both subplots.
plt.legend(fontsize=14)

plt.figure()
plt.clf()
plt.semilogy(pulseOxValueList, totalMeast_w, marker="D", markeredgecolor='k', \
    markerfacecolor='red', label="White", linewidth=0)
plt.semilogy(pulseOxValueList, totalMeast_b, marker="o", markeredgecolor='k', \
    markerfacecolor='purple', label="Black", linewidth=0)
plt.semilogy(pulseOxValueList, totalMeast_h, marker="*", markeredgecolor='k', \
    markerfacecolor='orange', label="Hispanic", linewidth=0)
plt.semilogy(pulseOxValueList, totalMeast_a, marker="X", markeredgecolor='k', \
    markerfacecolor='blue', label="Asian", linewidth=0)
plt.semilogy(pulseOxValueList, totalMeast_n, marker="s", markeredgecolor='k', \
    markerfacecolor='green',  label="Native Am.", linewidth=0)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Number of Measurements', fontsize=16)
plt.legend(fontsize=14)


plt.figure()
plt.clf()
plt.plot(pulseOxValueList, medianSaO2_w, 'rx', label="White", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_b, 'kx', label="Black", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_a, 'go', label="Asian", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_h, 'cv', label="Hispanic", linewidth=2)
plt.plot(pulseOxValueList, medianSaO2_n, 'b*', label="Native Am.", linewidth=2)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Median SaO2', fontsize=16)
plt.xlim([75,100])  
plt.ylim([75,100])  
plt.legend(fontsize=14)

plt.figure()
plt.clf()
plt.plot(pulseOxValueList, meanSaO2_w, marker="D", color='red', label="White", linewidth=0)
plt.plot(pulseOxValueList, meanSaO2_h, marker="*", color='orange', label="Hispanic", linewidth=0)
plt.plot(pulseOxValueList, meanSaO2_n, marker="s", color='green',  label="Native Am.", linewidth=0)
plt.plot(pulseOxValueList, meanSaO2_a, marker="X", color='blue', label="Asian", linewidth=0)
plt.plot(pulseOxValueList, meanSaO2_b, marker="o", color='purple', label="Black", linewidth=0)
plt.grid('on')
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Mean SaO2', fontsize=16)
plt.xlim([84.5,100])  
plt.ylim([83,100])  
plt.legend(fontsize=14)

# plt.figure()
# plt.clf()
# bins = len(pulseOxValueList)
# biasSpO2_w = [None]*bins
# biasSpO2_b = [None]*bins
# biasSpO2_a = [None]*bins
# biasSpO2_h = [None]*bins
# biasSpO2_n = [None]*bins
# for (i,SpO2) in enumerate(pulseOxValueList):
#     if meanSaO2_w[i] > 0:
#         biasSpO2_w[i] = pulseOxValueList[i] - meanSaO2_w[i] 
#     if meanSaO2_b[i] > 0:
#         biasSpO2_b[i] = pulseOxValueList[i] - meanSaO2_b[i] 
#     if meanSaO2_a[i] > 0:
#         biasSpO2_a[i] = pulseOxValueList[i] - meanSaO2_a[i] 
#     if meanSaO2_h[i] > 0:
#         biasSpO2_h[i] = pulseOxValueList[i] - meanSaO2_h[i] 
#     if meanSaO2_n[i] > 0:
#         biasSpO2_n[i] = pulseOxValueList[i] - meanSaO2_n[i]
# plt.plot(pulseOxValueList, biasSpO2_w, 'rx', label="White", linewidth=2)
# plt.plot(pulseOxValueList, biasSpO2_b, 'kx', label="Black", linewidth=2)
# plt.plot(pulseOxValueList, biasSpO2_a, 'go', label="Asian", linewidth=2)
# plt.plot(pulseOxValueList, biasSpO2_h, 'cv', label="Hispanic", linewidth=2)
# plt.plot(pulseOxValueList, biasSpO2_n, 'b*', label="Native Am.", linewidth=2)
# plt.grid('on')
# plt.xlabel('Pulse Ox Value (%)', fontsize=16)
# plt.ylabel('Bias (SpO2 - SaO2)', fontsize=16)
# plt.xlim([83.5,100])  
# plt.ylim([-5,5])
# plt.legend(fontsize=14)


# Plot box plots, which show some limited info about the distributions of SaO2 given SpO2.
plt.figure()
plt.clf()
medianprops = dict(color="black",linewidth=2)
bplot_w = plt.boxplot(biasSaO2_w[21:26],  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="red"), vert=True, patch_artist=True, positions=range(1,5*6+1,6))
bplot_h = plt.boxplot(biasSaO2_h[21:26],  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="orange"), vert=True, patch_artist=True, positions=range(2,5*6+2,6))
bplot_n = plt.boxplot(biasSaO2_n[21:26],  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="yellow"), vert=True, patch_artist=True, positions=range(3,5*6+3,6))
bplot_a = plt.boxplot(biasSaO2_a[21:26],  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="green"), vert=True, patch_artist=True, positions=range(4,5*6+4,6))
bplot_b = plt.boxplot(biasSaO2_b[21:26],  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="blue"), vert=True, patch_artist=True, positions=range(5,5*6+5,6))

sta_inds = [22, 27, 32]
end_inds = [27, 32, 36]
bgroup_w = [None]*3
bgroup_h = [None]*3
bgroup_n = [None]*3
bgroup_a = [None]*3
bgroup_b = [None]*3
for (i,sta) in enumerate(sta_inds):
    bgroup_w[i] = np.hstack(biasSaO2_w[sta:end_inds[i]])
    bgroup_h[i] = np.hstack(biasSaO2_h[sta:end_inds[i]])
    bgroup_n[i] = np.hstack(biasSaO2_n[sta:end_inds[i]])
    bgroup_a[i] = np.hstack(biasSaO2_a[sta:end_inds[i]])
    bgroup_b[i] = np.hstack(biasSaO2_b[sta:end_inds[i]])
plt.figure()
plt.clf()
medianprops = dict(color="black",linewidth=2)
bplot_w = plt.boxplot(bgroup_w,  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="red"), vert=True,  \
    patch_artist=True, positions=range(1,3*6+1,6))
bplot_h = plt.boxplot(bgroup_h,  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="orange"), vert=True,  \
    patch_artist=True, positions=range(2,3*6+2,6))
bplot_n = plt.boxplot(bgroup_n,  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="green"), vert=True,  \
    patch_artist=True, positions=range(3,3*6+3,6))
bplot_a = plt.boxplot(bgroup_a,  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="blue"), vert=True, \
    patch_artist=True, positions=range(4,3*6+4,6))
bplot_b = plt.boxplot(bgroup_b,  medianprops=medianprops, sym='', \
    boxprops=dict(facecolor="purple"), vert=True, \
    patch_artist=True, positions=range(5,3*6+5,6))
#plt.xticks([3,9,15], ['86-90', '91-95', '96-100'])
plt.xticks([3,9,15], ['87 to 91', '92 to 96', '97 to 100'])
plt.xlabel('Pulse Ox Value Range', fontsize=16)
plt.ylabel('SpO2 $-$ SaO2 (%) Box Plots', fontsize=16)
plt.grid(visible=True, axis='y')
plt.grid(visible=False, axis='x')
#plt.legend()
#y_range = [-15,23]
y_range = [-16,16]
plt.xlim([0, 18])
plt.ylim(y_range)
plt.plot([6,6],y_range,'k-', linewidth=0.5)
plt.plot([12,12],y_range,'k-', linewidth=0.5)
plt.legend([bplot_w["boxes"][0], bplot_h["boxes"][0], \
    bplot_n["boxes"][0], bplot_a["boxes"][0], bplot_b["boxes"][0]], \
    ['White', 'Hispanic', 'Native Am.', 'Asian', 'Black'], \
    loc='lower right', ncol=2, fontsize=14)




# Create a table of SpO2 bias by SpO2 Range and race/ethnicity for a LaTeX publication.
# Plot the same bias data.
plt.figure()
plt.clf()
i=0
meanBiasGroup_w = [None]*3
meanBiasGroup_h = [None]*3
meanBiasGroup_n = [None]*3
meanBiasGroup_a = [None]*3
meanBiasGroup_b = [None]*3
junk, meanBiasGroup_w[0] = plot_confidence_interval(1 , bgroup_w[0], marker="D", color='red', legendlabel="White")
junk, meanBiasGroup_h[0] = plot_confidence_interval(2 , bgroup_h[0], marker="*", color='orange', legendlabel="Hispanic")
junk, meanBiasGroup_n[0] = plot_confidence_interval(3 , bgroup_n[0], marker="s", color='green', legendlabel="Native Am.")
junk, meanBiasGroup_a[0] = plot_confidence_interval(4 , bgroup_a[0], marker="X", color='blue', legendlabel="Asian")
junk, meanBiasGroup_b[0] = plot_confidence_interval(5 , bgroup_b[0], marker="o", color='purple', legendlabel="Black")
for i in [1,2]:
    junk, meanBiasGroup_w[i] = plot_confidence_interval(1 + i*6, bgroup_w[i], marker="D", color='red')
    junk, meanBiasGroup_h[i] = plot_confidence_interval(2 + i*6, bgroup_h[i], marker="*", color='orange')
    junk, meanBiasGroup_n[i] = plot_confidence_interval(3 + i*6, bgroup_n[i], marker="s", color='green')
    junk, meanBiasGroup_a[i] = plot_confidence_interval(4 + i*6, bgroup_a[i], marker="X", color='blue')
    junk, meanBiasGroup_b[i] = plot_confidence_interval(5 + i*6, bgroup_b[i], marker="o", color='purple')
print("Pulse Ox Bias vs. Race/Ethnicity and SpO2 Range")
print("White & " 
    + "{:.2f}".format(meanBiasGroup_w[0]) + "\\% & " \
    + "{:.2f}".format(meanBiasGroup_w[1]) + "\\% & "
    + "{:.2f}".format(meanBiasGroup_w[2]) + "\\%  \\\\")
print("Black & " 
    + "{:.2f}".format(meanBiasGroup_b[0]) + "\\% & " \
    + "{:.2f}".format(meanBiasGroup_b[1]) + "\\% & "
    + "{:.2f}".format(meanBiasGroup_b[2]) + "\\%  \\\\")
print("Asian & " 
    + "{:.2f}".format(meanBiasGroup_a[0]) + "\\% & " \
    + "{:.2f}".format(meanBiasGroup_a[1]) + "\\% & "
    + "{:.2f}".format(meanBiasGroup_a[2]) + "\\%  \\\\")
print("Hispanic & " 
    + "{:.2f}".format(meanBiasGroup_h[0]) + "\\% & " \
    + "{:.2f}".format(meanBiasGroup_h[1]) + "\\% & "
    + "{:.2f}".format(meanBiasGroup_h[2]) + "\\%  \\\\")
print("Native American & " 
    + "{:.2f}".format(meanBiasGroup_n[0]) + "\\% & " \
    + "{:.2f}".format(meanBiasGroup_n[1]) + "\\% & "
    + "{:.2f}".format(meanBiasGroup_n[2]) + "\\%  \\\\")

#plt.xticks([1,2,3,4,5, 7,8,9,10,11, 13,14,15,16,17], \
#    ['','','87-91','','', '','','92-96','','','','', '97-100','',''])
plt.xticks([3,9,15], ['87 to 91', '92 to 96', '97 to 100'])
y_range = [-1.0,4.7]
plt.ylim(y_range)
plt.plot([6,6],y_range,'k-', linewidth=0.5)
plt.plot([12,12],y_range,'k-', linewidth=0.5)
plt.xlabel('Pulse Ox Measurement (%) Range', fontsize=16)
plt.ylabel('Statistical Bias, SpO2 $-$ SaO2 (%)', fontsize=16)
plt.legend(loc='lower right', ncol=2, fontsize=14)
#plt.ylim([-18,30])
plt.grid(visible=True, axis='y')
plt.grid(visible=False, axis='x')


# Calculate and bar plot the standard deviation of each group
sta_inds = [22, 27, 32]
end_inds = [27, 32, 36]
egroup_w = [None]*3
egroup_h = [None]*3
egroup_n = [None]*3
egroup_a = [None]*3
egroup_b = [None]*3
for (i,sta) in enumerate(sta_inds):
    egroup_w[i] = np.std(np.hstack(err_w[sta:end_inds[i]]))
    egroup_h[i] = np.std(np.hstack(err_h[sta:end_inds[i]]))
    egroup_n[i] = np.std(np.hstack(err_n[sta:end_inds[i]]))
    egroup_a[i] = np.std(np.hstack(err_a[sta:end_inds[i]]))
    egroup_b[i] = np.std(np.hstack(err_b[sta:end_inds[i]]))
plt.figure()
plt.clf()
bplot_w = plt.bar(range(1,3*6+1,6), egroup_w, color="red", label='White')
bplot_h = plt.bar(range(2,3*6+2,6), egroup_h, color="orange", label='Hispanic')
bplot_n = plt.bar(range(3,3*6+3,6), egroup_n, color="green", label='Native Am.')
bplot_a = plt.bar(range(4,3*6+4,6), egroup_a, color="blue", label='Asian')
bplot_b = plt.bar(range(5,3*6+5,6), egroup_b, color="purple", label='Black')
plt.xticks([3,9,15], ['87 to 91', '92 to 96', '97 to 100'])
plt.xlabel('Pulse Ox Value Range', fontsize=16)
plt.ylabel('Std. Deviation of SaO2 (%)', fontsize=16)
plt.grid(visible=True, axis='y')
plt.grid(visible=False, axis='x')
y_range = [0,14]
plt.xlim([0, 18])
plt.ylim(y_range)
plt.plot([6,6],y_range,'k-', linewidth=0.5)
plt.plot([12,12],y_range,'k-', linewidth=0.5)
plt.legend( loc='upper right', ncol=2, fontsize=14)

# Calculate and plot the median absolute deviation of each group
emgroup_w = [None]*3
emgroup_h = [None]*3
emgroup_n = [None]*3
emgroup_a = [None]*3
emgroup_b = [None]*3
for (i,sta) in enumerate(sta_inds):
    emgroup_w[i] = median_abs_deviation(np.hstack(errm_w[sta:end_inds[i]]))
    emgroup_h[i] = median_abs_deviation(np.hstack(errm_h[sta:end_inds[i]]))
    emgroup_n[i] = median_abs_deviation(np.hstack(errm_n[sta:end_inds[i]]))
    emgroup_a[i] = median_abs_deviation(np.hstack(errm_a[sta:end_inds[i]]))
    emgroup_b[i] = median_abs_deviation(np.hstack(errm_b[sta:end_inds[i]]))
plt.figure()
plt.clf()
bplot_w = plt.bar(range(1,3*6+1,6), emgroup_w, color="red", label='White')
bplot_h = plt.bar(range(2,3*6+2,6), emgroup_h, color="orange", label='Hispanic')
bplot_n = plt.bar(range(3,3*6+3,6), emgroup_n, color="green", label='Native Am.')
bplot_a = plt.bar(range(4,3*6+4,6), emgroup_a, color="blue", label='Asian')
bplot_b = plt.bar(range(5,3*6+5,6), emgroup_b, color="purple", label='Black')
plt.xticks([3,9,15], ['87 to 91', '92 to 96', '97 to 100'])
plt.xlabel('Pulse Ox Value Range', fontsize=16)
plt.ylabel('Median Absolute Deviation of SaO2 (%)', fontsize=16)
plt.grid(visible=True, axis='y')
plt.grid(visible=False, axis='x')
y_range = [0,4.5]
plt.xlim([0, 18])
plt.ylim(y_range)
plt.plot([6,6],y_range,'k-', linewidth=0.5)
plt.plot([12,12],y_range,'k-', linewidth=0.5)
plt.legend( loc='upper right', ncol=2, fontsize=14)


# Plot pmfs of the SaO2 value by SpO2 range and race
binedges = np.arange(-15.5, 16.5, 1.0)
bincenters = np.arange(-15,16,1)
pmf_group_w = [None]*3
pmf_group_h = [None]*3
pmf_group_n = [None]*3
pmf_group_a = [None]*3
pmf_group_b = [None]*3
for (i,sta) in enumerate(sta_inds):
    pmf_group_w[i], binsOut = np.histogram(np.hstack(biasSaO2_w[sta:end_inds[i]]), \
        binedges, density=True)
    pmf_group_h[i], binsOut = np.histogram(np.hstack(biasSaO2_h[sta:end_inds[i]]), \
        binedges, density=True)
    pmf_group_n[i], binsOut = np.histogram(np.hstack(biasSaO2_n[sta:end_inds[i]]), \
        binedges, density=True)
    pmf_group_a[i], binsOut = np.histogram(np.hstack(biasSaO2_a[sta:end_inds[i]]), \
        binedges, density=True)
    pmf_group_b[i], binsOut = np.histogram(np.hstack(biasSaO2_b[sta:end_inds[i]]), \
        binedges, density=True)
plt.figure()
plt.clf()
plt.subplots(3, 1, constrained_layout=False)
ax1 = plt.subplot(3,1,1)
plt.plot(bincenters, pmf_group_w[0], marker="D", markeredgecolor='k', \
    markerfacecolor='red', color='r', linewidth=1)
plt.plot(bincenters, pmf_group_b[0], marker="o", markeredgecolor='k', \
    markerfacecolor='purple', color='purple', linewidth=1)
plt.grid('on')
plt.text(5.5, 0.075, 'SpO2 87-91', fontsize=14)
ax2 = plt.subplot(3,1,2, sharex=ax1)
plt.plot(bincenters, pmf_group_w[1], marker="D", markeredgecolor='k', \
    markerfacecolor='red', color='r', linewidth=1)
plt.plot(bincenters, pmf_group_b[1], marker="o", markeredgecolor='k', \
    markerfacecolor='purple',  color='purple', linewidth=1)
plt.grid('on')
plt.text(5.5, 0.1, 'SpO2 92-96', fontsize=14)
ax3 = plt.subplot(3,1,3, sharex=ax2)
plt.plot(bincenters, pmf_group_w[2], marker="D", markeredgecolor='k', \
    markerfacecolor='red', label="White", color='r', linewidth=1)
plt.plot(bincenters, pmf_group_b[2], marker="o", markeredgecolor='k', \
    markerfacecolor='purple', label="Black", color='purple', linewidth=1)
plt.grid('on')
plt.text(5.5, 0.15, 'SpO2 97-100', fontsize=14)
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
plt.legend(fontsize=14, loc="upper left")
plt.xlabel('Error, SpO2 $-$ SaO2', fontsize=16)
plt.xlim([-15,15])
plt.text(-19.0, 0.15, 'Probability Mass Function', fontsize=16, rotation='vertical')



# What is the probability of extreme errors in SpO2 by race?

prob_large_error_w = ((errors_w < -10) | (errors_w > 10)).sum()/len(errors_w)
prob_large_error_b = ((errors_b < -10) | (errors_b > 10)).sum()/len(errors_b)
prob_large_error_a = ((errors_a < -10) | (errors_a > 10)).sum()/len(errors_a)
prob_large_error_h = ((errors_h < -10) | (errors_h > 10)).sum()/len(errors_h)
prob_large_error_n = ((errors_n < -10) | (errors_n > 10)).sum()/len(errors_n)
prob_large_error_all = ((errors_all < -10) | (errors_all > 10)).sum()/len(errors_all)
pvalue_bin_b = binomialTest(prob_large_error_w, prob_large_error_b, len(errors_w), len(errors_b))
pvalue_bin_a = binomialTest(prob_large_error_w, prob_large_error_a, len(errors_w), len(errors_a))
pvalue_bin_h = binomialTest(prob_large_error_w, prob_large_error_h, len(errors_w), len(errors_h))
pvalue_bin_n = binomialTest(prob_large_error_w, prob_large_error_n, len(errors_w), len(errors_n))
print("Probability of |SpO2-SaO2|>10, by Race/Ethnicity")
print("White & " + "{:.4f}".format(np.median(prob_large_error_w)) + " &  \\\\")
print("Black & " + "{:.4f}".format(np.median(prob_large_error_b)) + " &  *** \\\\")
print("Asian & " + "{:.4f}".format(np.median(prob_large_error_a)) + " &   \\\\")
print("Hispanic & " + "{:.4f}".format(np.median(prob_large_error_h)) + " & \\\\")
print("Native Am. & " + "{:.4f}".format(np.median(prob_large_error_n)) + " &  \\\\")
print("All Data & " + "{:.4f}".format(np.median(prob_large_error_all)) + " &  \\\\")



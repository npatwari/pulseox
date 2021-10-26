Assignment design: Neal Patwari, Oct. 2021

## Summary

Your goal for this assignment is to study the racial bias of pulse oximetry from a data set. Your data set was created to match the statistics of the [Sjoding 2020] measurements, as described in Figure 1 of their paper.  Your goal is to study the measurement bias as a function of race, and to see the impact that it has for the detection of hypoxemia.  We will study "race blind" use of pulse oximeter measurements, as well as "race corrected" measurements, for hypoxemia detection.  The description you need to understand my questions are in normal text, while the specific things I want you to implement are in bold, noted with a "Code This:" tag. Questions to answer in your write-up are denoted by an "Answer This:" tag.

## Submission

You should submit both your code and your write-up. While you can discuss problems with classmates, all of the work should be your own. You can submit your answers in a pdf file plus a code file, or in a jupyter notebook, whatever is easier for you.  Please label sections in both your code and write-up.

## A disclaimer

I generated this data to match the box plots shown in Figure 1.  This is not the real data from the two data sets used by the authors, which is unavailable to me.  Instead, I generated pairs of (pulse ox, arterial oxygen saturation) data that matches the characteristics of the data reported in the box plots in Figure 1.  You can access my code for generating fake (but matching the data in the figure) in the github project mentioned below if you want to know more. It uses a metalog distribution to ensure that the median, 25th percentile, and 75th percentile of the generated arterial oxygen saturation measurements match those in the plots.  They aren't, though, the original data, and so your results have this caveat.

## Assignment Tasks and Instructions:

### 1. Load and understand the data. 

The data and some introductory code is at: 

https://github.com/npatwari/pulseox

**Code This**: Use numpy's `loadtxt()` command (or Matlab's `csvread()`) to load the two CSV files. Separate out the pulse oximeter data in column 0 (note my column numbering starts at 0, if you use Matlab you start at 1), and the arterial oxygen saturation data in column 1.

My scripts `plotPulseOxData.py` and `plotPulseOxData.m` contain the python / Matlab code to do this (and a little more).

The 0th column is the pulse ox value:
- This is what we are directly studying.  We are considering an application in which we only have pulse ox values and need to detect hypoxemia.  (In this data set, we also have a arterial oxygen saturation, but this is only to be used for our assignment as a ground truth.)

The 1st column is the arterial oxygen saturation:
- We take the arterial Ox Sat as the "truth" because it is the medical "gold standard" for monitoring of oxygen saturation in the blood.

Each row is one patient:  
- The two measurements were taken within 10 minutes of each other, so the authors consider them to be comparable [Sjoding 2020].
 
Our two hypotheses are H0 and H1:
- H0: the "normal" case in which the patient does not have hypoxemia    
- H1: the "abnormal", what we want to be alarmed about, the patient has hypoxemia

The ground truth about H0 (and H1) is known to us, it is whenever the Arterial Oxygen Saturation is >= 88.0 (or < 88.0 in the case of H1).  But we are studying a system that only has the pulse oximeter value.  We need to set a threshold on the pulse oximeter value, and if it is less than the threshold we'll decide H1; and if it is larger than the threshold we will decide H0.

### 2. Be able to calculate Prob[ Correct Detection ] and Prob[ False Alarm ].

There are two ways to make a correct decision. These are:
1. Correct Detection (aka True Positive): We decide H1 when in fact the patient has hypoxemia
2. True Negative: We decide H0 when in fact the patient does not have hypoxemia

Each of the above has a probability or frequency of occurrence:
1. Prob[ Correct Detection ] = # of correct decisions in the data set / # of hypoxemia cases in the data set
2. Prob[ True Negative ] = # of true negative decisions in the data set / # of cases in the data set that are not hypoxemia

There are also two ways to make an incorrect decision. These are:
1. False Alarm (aka Type I Error): We decide H1 when in fact the patient does not have hypoxemia
2. False Negative (aka Type II error): We decide H0 when in fact the patient has hypoxemia

Each of the above has a probability or frequency of occurance:
1. Prob[ False Alarm ] = # of false alarms in the data set / # of cases in the data set that are not hypoxemia
2. Prob[ False Negative ] = # of false negative decisions in the data set / # of hypoxemia cases in the data set

Note that we make a decision no matter what the pulse ox value is.  So:
- Prob[ Correct Detection ] + Prob[ False Negative ] = 1
- Prob[ True Negative ] + Prob[ False Alarm ] = 1

That is, there are really two metrics of performance of a detector.  If you know the Prob[ Correct Detection ] and the Prob[ False Alarm ], for example, you will know all four.

The two types of incorrect decisions might lead to different negative outcomes.  Let's take the example of a patient with covid who uses a pulse oximeter at home to decide if they should go to the ER.
- A false alarm could lead someone to go to the ER, which could be costly and take their time, when they didn't really need the intensive medical care to survive the virus.
- A false negative would keep them at home at a time when they might need supplemental oxygen or other treatment to survive the virus, and thus would increase their risk of death.

**Code This:** For a threshold of 91.5, calculate the probability of false alarm, and probability of correct detection when using a pulse ox value to detect hypoxemia.  Do this separately for Black and white patients. Finally, calculate the probability of false alarm, and probability of detection over ALL patients. (This is not the average of the Black and white values because there are a different number of measurements for Black patients and white patients).  

2) **Answer This:** Turn in the six probability values (probability of false alarm and probability of correct detection; for Black, white, and all patients).

### 3. Calculate and plot the results for all possible thresholds.

You might be saying, the performance metrics are a function of the threshold.  As you move the threshold, one type of error increases while the other decreases. Our motivating question for this section is, what should the threshold be?

There are only integer values from the pulse oximeters, and the data we are given includes only values 89 to 96.  So there are only 8 thresholds that would give us different results.  So let's study the range from 88.5 to 96.5 with step size 1.0.  

**Code This:** For each threshold, calculate the six probabilities you did in part 2.  Put them in six numpy arrays.  For your reference, in my code I called them:

| | Prob[False Alarm]  |  Prob[Correct Detection] |
|--------|-------------|--------------------------|
| white: |   `p_FA_w`    |        `p_CD_w`        |
| Black: |   `p_FA_b`    |        `p_CD_b`        |
| All:   |   `p_FA_all`  |        `p_CD_all`      |




Engineers use what is called a "receiver operating characteristic" (ROC) curve to study the tradeoff in error types for any detector.  This is simply a plot of the Prob[ Correct Detection ] as a function of the Prob[ False Alarm ], for all possible thresholds.  

3a) **Code and Answer This**: Plot each (probability of false alarm, probability of detection) pair as a point on a figure.  Label them by race, and with the threshold.  Turn in the plot.

You can use my python code to plot the values.
```
# Plot the results
plt.figure(1)
plt.plot(p_FA_w, p_CD_w, 'rs', label="White", linewidth=2)
plt.plot(p_FA_b, p_CD_b, 'ko', label="Black", linewidth=2)
plt.plot(p_FA_all, p_CD_all, 'g.', label="All", linewidth=2)
plt.grid('on')
plt.xlabel('Probability of False Alarm', fontsize=16)
plt.ylabel('Probability of Correct Detection', fontsize=16)
plt.xticks(np.arange(0, 1.01, 0.1)) 
plt.yticks(np.arange(0, 1.01, 0.1))
plt.legend(fontsize=16)
for i, threshold in enumerate(threshold_list):

    # Put the threshold on each dot, connect the white/Black points for 
    # that correspond to the same threshold.
    plt.text(p_FA_w[i], p_CD_w[i], str(threshold), horizontalalignment='right')
    plt.text(p_FA_b[i], p_CD_b[i], str(threshold), horizontalalignment='left')
    plt.plot([p_FA_b[i],p_FA_w[i]], [p_CD_b[i],p_CD_w[i]], 'b-', linewidth=2)
```

3b) **Answer This**: From this data, what would be the best threshold to minimize:
1. the sum of the two probabilities of error (Prob[ False Alarm ] + Prob[ False Negative]), if considering all patients together?
2. the sum of the two probabilities of error, if considering only white patients?
3. the sum of the two probabilities of error, if considering only Black patients?


## 4.  Consider having a different threshold by race.  

This would be, essentially, a race-based correction factor.  Think about this: Having a different threshold as a function of race is the same as adding or subtracting some race-based constant from the pulse oximeter value (to correct for the offset), and then having a race-neutral threshold.

4) **Answer This**:  Is there a way to set a different threshold by race, in this data set, such that the performance of the hypoxemia detector is identical for white and Black patients?  Exclude from consideration the threshold < 89 (which never detects hypoxemia) and the threshold > 96 (which always detects hypoxemia).

## 5. Your Ideas: What is the deeper problem?

You should see now that, even when using a race-based correction factor, the performance of a pulse oximeter for detecting hypoxemia is worse for patients who are Black than for patients who are white.  

5) **Answer This**: In 1-2 sentences, if the statistical bias in the pulse ox value as a function of race is not the problem when the detector is corrected for race, what do you think is the problem that causes the hypoxemia detector to perform worse for Black patients?  Brainstorming is ok for this -- I don't expect you to analyze the data to justify your ideas, and it will not be graded for correctness (only as answered/unanswered).



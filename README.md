# NN_classification_starter_kit
Simple project starter kit for 2022-2R APPLICATIONS AND PRACTICE IN NEURAL NETWORKS

# Reference 
- dataset : https://www.kaggle.com/datasets/wanghaohan/confused-eeg
- information : http://www.cs.cmu.edu/~kkchang/paper/WangEtAl.2013.AIED.EEG-MOOC.pdf



# Data Information
## Data
- name : EEG data from 10 students watching MOOC videos
- for : Can EEG detect confusion?
- Experimental settings
    - 10-20 system
    - An experiment with a student consisted of 10 sessions.
    - prepared 20 videos, 10 in each category. Each video was about 2 minutes long (We chopped the two-minute clip in the middle of a topic to make the videos more confusing)
    - randomly picked five videos of each category and randomized the presentation sequence so that the student could not guess the predefined confusion level.
    - the student was first instructed to relax their mind for 30 seconds
    - After each session, the student rated his/her confusion level on a scale of 1-7, where 1 corresponded to the least confusing and 7 corresponded to the most confusing
    - there were three student observers watching the body-language of the student. Each observer rated the confusion level of the student in each session on a scale of 1-7.
    1. The raw EEG signal, sampled at 512 Hz
    2. An indicator of signal quality, reported at 1 Hz
    3. MindSet’s proprietary “attention” and “meditation” signals said to measure
    the user’s level of mental focus and calmness, reported at 1 Hz
    4. A power spectrum, reported at 8 Hz, clustered into the standard named frequency bands: delta (1-3Hz), theta (4-7 Hz), alpha (8-11 Hz), beta (12-29
    Hz), and gamma (30-100 Hz).
   
## N_timepoints per subject

SubjectID
0.0    1261 (timepoints)
1.0    1301
2.0    1284
3.0    1314
4.0    1295
5.0    1262
6.0    1275
7.0    1276
8.0    1282
9.0    1261


## Feature description

### about subjects

**SubjectID**

- participant identification number
- 0~9

**VideoID**

- Video numbers played to participants
- 0~9

###about EEG signal

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6cc5ec4c-7f3f-482a-a629-3dd0b3b7d610/Untitled.png)

**Attention**

- **Mindset** (Attention and Mediation) : The students wore a single-channel wireless MindSet that measured activity over the frontal lobe. The MindSet measures the voltage between an electrode resting on the forehead and two electrodes (one ground and one reference) each in contact with an ear. More precisely, the position on the forehead is Fp1 (somewhere between left
eye brow and the hairline), as defined by the International 10-20 system [12].
- Average proprietary measure of mental focus (sampling rate : 1 Hz)
- 0~100 (integer)

**Mediation**

- Average proprietary measure of calmness (sampling rate : 1 Hz)
- 0~100 (integer)

**Raw**

- mean of Raw EEG signal amplitude per 0.5 sec (sampling rate :512Hz)
- each data point consists of 120+ rows, which is sampled every 0.5 seconds
- -2.5k ~ 2.5k (float)

**Delta**

- 1-3 Hz of power spectrum density per 0.5sec (sampling rate :8Hz)
- 

**Theta**

- 4-7 Hz of power spectrum density per 0.5sec (sampling rate :8Hz)

**Alpha1**

- Lower 8-11 Hz of power spectrum density per 0.5sec (sampling rate :8Hz)

**Alpha2**

- Higher 8-11 Hz of power spectrum per 0.5sec (sampling rate :8Hz)

**Beta1**

- Lower 12-29 Hz of power spectrum density per 0.5sec (sampling rate :8Hz)

**Beta2**

- Higher 12-29 Hz of power spectrum density per 0.5sec (sampling rate : 8Hz)

**Gamma1**

- Lower 30-100 Hz of power spectrum density per 0.5sec (sampling rate : 8Hz)

**Gamma2**

- Higher 30-100 Hz of power spectrum density per 0.5sec (sampling rate : 8Hz)

### about Labels

**predefined confusion level**

- confusion level according to the experiment design (whether the subject is expected to be confused)
- binary class (0 : doesn’t seem to be confused, 1 : seems to be confused)

**userdefined confusion level**

- confusion level according to each user’s subjective rating (whether the subject is actually confused)
- binary class (0 : not confused, 1 :  confused)

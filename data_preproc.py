#%%
import scipy.io as sio
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
import scipy.signal as spsg
from scipy.optimize import curve_fit
from scipy.stats import skew
from meegkit.dss import dss_line
import datetime
from sklearn.decomposition import FastICA
import numpy.matlib as matlib
from math import floor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

lfp1_data = np.load('../lfp1_data.npy')
lfp2_data = np.load('../lfp2_postprocess.npy')
with open('../meta.pkl', 'rb') as fp:
    meta = pickle.load(fp)

ratio = meta['ratio']
freq = meta['fs']
# -----------------------------------------------------------------------------
# This is the event data, you may change this to change the classification task
params = sio.loadmat('../Params_2018_06_06_Y.mat')["TrialsParameters"]
rh_indexes = None
with open('../motorno.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        rh_indexes = np.array(row)

rh_indexes = np.where(rh_indexes != 'all')

CueOn = params['TimeCueOn_samples'][0][0][0][rh_indexes]
CueOff = params['TimeCueOff_samples'][0][0][0][rh_indexes]
GraspStart = params['TimeGraspStart_samples'][0][0][0][rh_indexes]
GraspMax = params['TimeGraspMax_samples'][0][0][0][rh_indexes]
TrialEnd = params['TimeReward_samples'][0][0][0][rh_indexes]
GraspEnd = params['TimeGraspEnd_samples'][0][0][0][rh_indexes]

TimeExit = params['TimeExitHP_samples'][0][0][0][rh_indexes]
TrialEndFull = params['TimeReward_samples'][0][0][0]
#%%
# -----------------------------------------------------------------------------

structured_data = [[[] for _ in range(5)] for _ in range(len(CueOn))] # trial x classes x channels x time
basefs = 4.8828125*10**3
discarded = set()

brain_areas = {"PMvR": (96, 96), "M1": (128, 128), "PMdR": (224, 224), "PMdL": (256, 256)}
back_mapping = [x for x in range(256)]
"""
# In case you want to load previously processed data
with open('brain_areas.pkl', 'rb') as fp:
    brain_areas = pickle.load(fp)
with open('back_mapping.pkl', 'rb') as fp:
    back_mapping = pickle.load(fp)
"""

#%%
#======================================================================
# Plotting stuff
#======================================================================
import matplotlib.ticker as ticker
fig, ax = plt.subplots()
power, freq_=plt.psd(lfp2_data[1], Fs=freq)
plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5, alpha=1)
plt.semilogy(freq_[256:-256], power[256:-256])
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))

plt.show()
#%%
plt.psd(lfp1_data[back_mapping[33]], Fs=freq)
plt.show()
#%%
plt.psd(lfp2_data[back_mapping[12+128]-128], Fs=freq)
plt.show()
#%%
from matplotlib.pyplot import specgram
istart = int(CueOn[84]//ratio)
istop = int(GraspEnd[84]//ratio)
cind = back_mapping[12+128]-128
powerSpectrum, freqenciesFound, time, imageAxis = specgram(lfp2_data[cind][istart:istop], Fs=int(freq))
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.show()   
#%%
# Power Spectral Densitity plot
for x in lfp2_data[:back_mapping[96+128]-128]:
    plt.psd(x, Fs=freq)
plt.title("PMdR Power Spectral Density Plot for all channels")
plt.show()
#%%
# Power Spectral Densitity plot
for x in lfp2_data:
    plt.psd(x, Fs=freq)
# function to show the plot
plt.show()

#%%
"""
---------------------------------------------------------------
PREPROCESSING
---------------------------------------------------------------
"""

#%%

############################################################
#                         Bandpass                         #
############################################################
def bandpass_filtering(lfp, frq, lp=1., hp=500.):
    b,a = spsg.iirfilter(3, [lp/frq,hp/frq], btype='bandpass', ftype='butter')
    for i, d in enumerate(lfp):
        print(i, end=" ")
        lfp[i] = spsg.filtfilt(b, a, d)

#%%
print("Bandpassing...")
print("\tProcessing LFP1 data...")
bandpass_filtering(lfp1_data, freq)
#%%
print("\n\tProcessing LFP2 data...")
bandpass_filtering(lfp2_data, freq)

#%%
############################################################
#                        ZapFilter                         #
############################################################
def zap_filtering(lfp,frq):
    lfp = np.transpose(lfp, (1, 0))
    lfp = lfp.reshape((lfp.shape[0], lfp.shape[1], 1))
    lfp = lfp.astype(np.float64)
    
    lfp, _ = dss_line(lfp, fline=60.,sfreq=frq,blocksize=8192)

    lfp = lfp.reshape((lfp.shape[0], lfp.shape[1]))
    lfp = np.transpose(lfp, (1, 0))
    return lfp.astype(np.int16)
#%%
print("ZapFiltering...")
print(datetime.datetime.now())
print("\tProcessing LFP1 data...")
lfp1_data = np.transpose(lfp1_data, (1, 0))
lfp1_data = lfp1_data.reshape((lfp1_data.shape[0], lfp1_data.shape[1], 1))
lfp1_data = lfp1_data.astype(np.float64)
#%%
lfp1_data, _ = dss_line(lfp1_data, fline=60.,sfreq=freq,blocksize=8192)
#%%
lfp1_data = lfp1_data.reshape((lfp1_data.shape[0], lfp1_data.shape[1]))
lfp1_data = np.transpose(lfp1_data, (1, 0))
lfp1_data = lfp1_data.astype(np.int16)
print(datetime.datetime.now())

#%%
np.save('lfp1_zap', lfp1_data)
#%%
#LFP2
print(datetime.datetime.now())
print("\tProcessing LFP2 data...")
lfp2_data = np.transpose(lfp2_data, (1, 0))
lfp2_data = lfp2_data.reshape((lfp2_data.shape[0], lfp2_data.shape[1], 1))
lfp2_data = lfp2_data.astype(np.float64)
#%%
lfp2_data, _ = dss_line(lfp2_data, fline=60.,sfreq=freq,blocksize=8192)
#%%
lfp2_data = lfp2_data.reshape((lfp2_data.shape[0], lfp2_data.shape[1]))
lfp2_data = np.transpose(lfp2_data, (1, 0))
lfp2_data = lfp2_data.astype(np.int16)
print(datetime.datetime.now())

#%%
np.save('lfp2_zap', lfp2_data)
#%%
############################################################
#                  Bad Channel Rejection                   #
############################################################
# Automatic Iterative Standard Deviation method (Komosar, et al. 2022)
def std_bad_channels(lfp):
    all_channels = np.arange(lfp.shape[0])
    remaining_channels = all_channels.copy()
    k = 0  # iteration counter
    sd_pk = np.inf  # std of all individual channel std's
    std_all = np.std(lfp, axis=1)
    while sd_pk > 5:
        sd_k = std_all[remaining_channels] # std of each channel
        m_k = np.median(sd_k)  # median of channel std's
        third_quartile = np.percentile(sd_k, 75)
        temp = np.std(sd_k)
        if sd_pk == temp:  # if no channels are removed (not in paper)
            break
        sd_pk = temp
        bad_channels_k = []
        for ch in remaining_channels:
            sd_jk = std_all[ch]
            if sd_jk < 10e-1:
                bad_channels_k.append(ch)
            elif sd_jk > 100:
                bad_channels_k.append(ch)
            elif abs(sd_jk - m_k) > third_quartile:
                bad_channels_k.append(ch)
        remaining_channels = np.setdiff1d(remaining_channels, bad_channels_k)
        k+=1
    bad_channels = np.setdiff1d(all_channels, remaining_channels)
    print("\trejecting channels:", bad_channels)
    return remaining_channels, bad_channels

def process_indexes(bdict, indexes, offset=0):
    for i in indexes:
        for x in bdict:
            if i+offset < bdict[x][0]:
                bdict[x] = (bdict[x][0], bdict[x][1]-1)
            
def process_backmap(mapping, indexes, offset=0):
    for i in indexes:
        mapping[i+offset] = None
        for j in range(i+offset+1,len(mapping)):
            if mapping[j]:
                mapping[j] -= 1

#%%

print(brain_areas)
print("Bad Channel Rejection")
print("\tProcessing LFP1 data...")
remaining_channels, bad_channels = std_bad_channels(lfp1_data)
process_backmap(back_mapping, bad_channels)
process_indexes(brain_areas, bad_channels)
lfp1_data = lfp1_data[remaining_channels]
print(brain_areas)
print(back_mapping)
#%%
print("\tProcessing LFP2 data...")
remaining_channels, bad_channels = std_bad_channels(lfp2_data)
process_backmap(back_mapping, bad_channels, offset=128)
process_indexes(brain_areas, bad_channels, offset=128)
lfp2_data = lfp2_data[remaining_channels]
print(brain_areas)
print(back_mapping)

#%%
############################################################
#                           ICA                            #
############################################################
def perform_ICA(eeg_d, start, end, random_state=None, save=""):
    # PMvR
    eeg_data = eeg_d[start:end,:]
    eeg_data = np.transpose(eeg_data)
    n_channels = eeg_data.shape[1]

    # Create an instance of ICA
    ica = FastICA(n_components=n_channels, random_state=random_state)

    # Fit the ICA model to your data
    eeg_d_t = ica.fit_transform(eeg_data)
    if save:
        with open(save + '_ica.pkl', 'wb') as fp:
            pickle.dump(ica, fp)
    return ica, eeg_d_t

#%%
############################################################
#                          MARA                            #
############################################################
def mara(ica, eeg_d, freq):
    print("MARA...")
    patterns = ica.mixing_
    patterns = patterns/matlib.repmat(np.std(patterns,axis=0), patterns.shape[0], 1)
    features = [[0 for _ in range(patterns.shape[0])] for _ in range(6)]

    print("\tComputing features...")
    # =========================================================
    # Compute spatial range
    # =========================================================
    spatial_range = np.log(patterns.max(axis=0)-patterns.min(axis=0))

    features[0] = spatial_range

    # =========================================================
    # time and frequency features
    # =========================================================

    eeg_d = np.transpose(eeg_d)
    # downsample to 100-200Hz
    factor = max(floor(freq/100),1)
    eeg_data_ds = eeg_d[:,0:eeg_d.shape[1]:factor]
    freq_ds = round(freq/factor)

    icacomps = (np.dot(ica.components_,eeg_data_ds)).conj().T
    icacomps = icacomps/matlib.repmat(np.std(icacomps,axis=0), icacomps.shape[0], 1)
    icacomps = icacomps.conj().T


    def myfun(xdata, *x):
        return np.exp(x[0])/np.power(xdata, np.exp(x[1])) - x[2]

    for ic in range(icacomps.shape[0]): # icacomps.shape[0]
        print(".",end="")
        # =========================================================
        # Proc spectrum for channel
        # =========================================================
        frq, pxx = spsg.welch(icacomps[ic],window=[1]*freq_ds,noverlap=None,fs=freq_ds)
        pxx = 10*np.log10(pxx * freq_ds/2)

        # =========================================================
        # Average log band power between 8 and 13Hz
        # =========================================================
        p = 0
        for i in range(8,14):
            p += pxx[np.where(frq == i)[0][0]]
        Hz8_13 = p / (13-8+1)

        # =========================================================
        # lambda and FitError: deviation of a component's spectrum from
        # a protoptypical 1/frequency curve 
        # =========================================================
        # first point
        p1 = [0,0]
        p1[0] = 2
        p1[1] = pxx[np.where(frq == p1[0])[0][0]]
        # second point
        p2 = {}
        p2[0] = 3
        p2[1] = pxx[np.where(frq == p2[0])[0][0]]
        # third point
        p3 = {}
        p3[1] = pxx[np.where(frq == 5)[0][0]:np.where(frq == 13)[0][0]].min()
        p3[0] = frq[np.where(pxx == p3[1])[0][0]]
        # fourth point
        p4 = {}
        p4[0] = p3[0] - 1
        p4[1] = pxx[np.where(frq == p4[0])[0][0]]
        # fifth point
        p5 = {}
        p5[1] = pxx[np.where(frq == 5)[0][0]:np.where(frq == 13)[0][0]].min()
        p5[0] = frq[np.where(pxx == p5[1])[0][0]]
        # sixth point
        p6 = {}
        p6[0] = p5[0] + 1
        p6[1] = pxx[np.where(frq == p6[0])[0][0]]

        pX = [p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]]
        pY = [p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]]
        xstart = [4, -2, 54]
        popt, pcov = curve_fit(myfun, pX, pY, p0=xstart, method='trf', maxfev=1000000)
        
        #FitError: mean squared error of the fit to the real spectrum in the band 2-40 Hz.
        ts_8to15 = frq[np.where(frq == 8)[0][0]:np.where(frq == 16)[0][0]]
        fs_8to15 = pxx[np.where(frq == 8)[0][0]:np.where(frq == 16)[0][0]]
        fiterror = np.log(np.power(np.linalg.norm(myfun(ts_8to15, *tuple(popt.tolist()))-fs_8to15),2))
        # lambda: parameter of the fit
        lambda_par = popt[1]

        # =========================================================
        # Averaged local skewness 15s
        # =========================================================
        interval = 15
        abs_local_skewness = []
        mean_abs_local_skewness_15 = 0

        for i in range(0,icacomps.shape[1]//freq_ds-interval+1,interval):
            fstart = i*freq_ds
            fstop = (i+interval)*freq_ds+1
            abs_local_skewness.append(np.abs(skew(icacomps[ic, fstart:fstop])))
        if not abs_local_skewness:
            print("bad")
        else:
            mean_abs_local_skewness_15 = np.log(np.mean(abs_local_skewness))

        features[1][ic] = mean_abs_local_skewness_15
        features[2][ic] = lambda_par
        features[3][ic] = Hz8_13 # type: ignore
        features[4][ic] = fiterror
    print("\nCompleted")
    return np.array(features).T
#%%
############################################################
#                       ClusterMARA                        #
############################################################

# PMvR
ica1, eeg_data_t = perform_ICA(lfp1_data, 0, brain_areas["PMvR"][1], random_state=42, save="PMvR")
np.save('PMvR', eeg_data_t)
# M1
ica2, eeg_data_t_2 = perform_ICA(lfp1_data, brain_areas["PMvR"][1], brain_areas["M1"][1], random_state=42, save="M1")
np.save('M1', eeg_data_t_2)
# PMdR
ica3, eeg_data_t_3 = perform_ICA(lfp2_data, 0, brain_areas["PMdR"][1] - brain_areas["M1"][1], random_state=42, save="PMdR")
np.save('PMdR', eeg_data_t_3)
# PMdL
ica4, eeg_data_t_4 = perform_ICA(lfp2_data, brain_areas["PMdR"][1] - brain_areas["M1"][1], brain_areas["PMdL"][1] - brain_areas["M1"][1], random_state=42, save="PMdL")
np.save('PMdL', eeg_data_t_4)
#%%
eeg_data_t = np.load('../PMvR.npy')
with open('../PMvR_ica.pkl', 'rb') as fp:
    ica1 = pickle.load(fp)
eeg_data_t_2 = np.load('../M1.npy')
with open('../M1_ica.pkl', 'rb') as fp:
    ica2 = pickle.load(fp)
eeg_data_t_3 = np.load('../PMdR.npy')
with open('../PMdR_ica.pkl', 'rb') as fp:
    ica3 = pickle.load(fp)
eeg_data_t_4 = np.load('../PMdL.npy')
with open('../PMdL_ica.pkl', 'rb') as fp:
    ica4 = pickle.load(fp)
#%%
# Calculate mara features
PMvR_mara = mara(ica1, eeg_data_t, freq)
M1_mara = mara(ica2, eeg_data_t_2, freq)
PMdR_mara = mara(ica3, eeg_data_t_3, freq)
PMdL_mara = mara(ica4, eeg_data_t_4, freq)
#%%
# MARA Clustering feature rejection
print("Clustering...")
# PMvR
print("\tPMvR")
clustering = DBSCAN(eps=2, min_samples=2)
fitted = clustering.fit(PMvR_mara)
rejected = np.where(fitted.labels_ == -1)[0]
eeg_data_t[:, rejected] = 0
ica1.inverse_transform(eeg_data_t, copy=False)
# M1
print("\tM1")
clustering = DBSCAN(eps=2, min_samples=2)
fitted = clustering.fit(M1_mara)
rejected = np.where(fitted.labels_ == -1)[0]
eeg_data_t_2[:, rejected] = 0
ica2.inverse_transform(eeg_data_t_2, copy=False)
# PMdR
print("\tPMdR")
clustering = DBSCAN(eps=2, min_samples=2)
fitted = clustering.fit(PMdR_mara)
rejected = np.where(fitted.labels_ == -1)[0]
eeg_data_t_3[:, rejected] = 0
ica3.inverse_transform(eeg_data_t_3, copy=False)
# PMdL
print("\tPMdL")
clustering = DBSCAN(eps=2, min_samples=2)
fitted = clustering.fit(PMdL_mara)
rejected = np.where(fitted.labels_ == -1)[0]
eeg_data_t_4[:, rejected] = 0
ica4.inverse_transform(eeg_data_t_4, copy=False)
#%%
lfp1_data = np.hstack([eeg_data_t, eeg_data_t_2]).T
lfp2_data = np.hstack([eeg_data_t_3, eeg_data_t_4]).T
print(lfp1_data.shape, lfp2_data.shape)

#%%

"""
---------------------------------------------------------------
END OF PREPROCESSING
---------------------------------------------------------------
"""
#==============================
#       MARA Plotting
#==============================
#%%
# show EEGs
fig, ax = plt.subplots(nrows=8,facecolor='none')
plt.subplots_adjust(hspace=0.2)  # Adjust the vertical spacing between subplots

# Remove background and axes from each subplot
for i in range(8):
    ax[i].axis('off')

values = {2: "tab:blue", 13: "tab:orange", 33:"tab:green", 49: "tab:red", 65: "tab:purple", 79: "tab:brown", 82: "tab:pink", 84: "tab:olive"}
# Plot the data with black lines
c = 0
for x in values:
    ax[c].plot(eeg_data_t.T[x,GraspStart[33]:GraspEnd[33]],color=values[x], linewidth=0.5)
    c += 1

# Adjust the figure size to make the plot more flat
fig.set_size_inches(8, 8)  # Increase the width and reduce the height as per your preference

# Show the plot
plt.show()
#%%
# Scatter plot
fig, ax = plt.subplots(figsize=(8, 8), sharey=True)
names = ["Spatial Range", "Alpha Log Band Power", "Lambda", "FitError", "Average Local Skewness 15s"]

c = 0
for x in values:
    ax.scatter(names, PMvR_mara[x,:5],color=values[x])    
c += 1
fig.autofmt_xdate(rotation=45)
plt.show()

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mara_data =PMvR_mara[:,:5]

pca = PCA()

principalComponents = pca.fit_transform(mara_data)
#%%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Clusters of MARA Features', fontsize = 20)

colors = ['r', 'g', 'b', 'c', 'm']
for target, color in zip(np.unique(fitted.labels_), colors):
    indexes_class = np.where(fitted.labels_ == target)
    ax.scatter(principalComponents[indexes_class,0]
               , principalComponents[indexes_class,1]
               , c = color
               , s = 50)
ax.legend(np.unique(fitted.labels_))
ax.grid()
#%%


#==============================
#          Epoching
#==============================

timesteps = 300
structured_data = [[[] for _ in range(5)] for _ in range(len(CueOn))] # trial x classes x channels x time
for i in range(len(CueOn)):
    templs = list()
    cOn = CueOn[i]
    cOff = CueOff[i]
    gStart = GraspStart[i]
    gMax = GraspMax[i]
    tEnd = TrialEnd[i]
    gEnd = GraspEnd[i]
    
    # prereach
    fstart = int(cOn//ratio)
    fend = int(cOff//ratio)
    duration = fend-fstart
    midpoint = (fend-fstart)//2
    fend = fstart + midpoint + timesteps//2
    fstart = fstart + midpoint - timesteps//2
    structured_data[i][0] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration <= timesteps:
        discarded.add(i)
    # reach
    fstart = int(cOff//ratio)
    fend = int(gStart//ratio)
    duration = fend-fstart
    midpoint = (fend-fstart)//2
    fend = fstart + midpoint + timesteps//2
    fstart = fstart + midpoint - timesteps//2
    structured_data[i][1] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration <= timesteps:
        discarded.add(i)
    # grasp max
    fstart = int(gStart//ratio)
    fend = int(gMax//ratio)
    duration = fend-fstart
    fend = fstart + midpoint + timesteps//2
    fstart = fstart + midpoint - timesteps//2
    structured_data[i][2] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration <= timesteps:
        discarded.add(i)
    # reward
    fstart = int(gMax//ratio)
    fend = int(tEnd//ratio)
    duration = fend-fstart
    fend = fstart + midpoint + timesteps//2
    fstart = fstart + midpoint - timesteps//2
    structured_data[i][3] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration <= timesteps:
        discarded.add(i)
    # release
    fstart = int(tEnd//ratio)
    fend = int(gEnd//ratio)
    duration = fend-fstart
    fend = fstart + midpoint + timesteps//2
    fstart = fstart + midpoint - timesteps//2
    structured_data[i][4] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration <= timesteps:
        discarded.add(i)

for x in sorted(list(discarded),reverse=True):
    structured_data.pop(x)

structured_data = np.array(structured_data)
print(structured_data.shape)
np.save('structured_data_full_5classes_300', structured_data)

# %%

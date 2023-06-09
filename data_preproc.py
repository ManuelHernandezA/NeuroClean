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
#%%
#lfp1_data = np.load('lfp1_data.npy')
lfp1_data = np.load('lfp1_preICA.npy')
#lfp1_data = np.load('lfp1_zap_v2.npy')
#%%
#lfp2_data = np.load('lfp2_data.npy')
lfp2_data = np.load('lfp2_preICA.npy')
#lfp2_data = np.load('lfp2_zap.npy')
#%%

with open('meta.pkl', 'rb') as fp:
    meta = pickle.load(fp)

ratio = meta['ratio']
freq = meta['fs']
#%%
params = sio.loadmat('Params_2018_06_06_Y.mat')["TrialsParameters"]
rh_indexes = None
with open('motorno.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        rh_indexes = np.array(row)

rh_indexes = np.where(rh_indexes == '4')

CueOn = params['TimeCueOn_samples'][0][0][0][rh_indexes]
CueOff = params['TimeCueOff_samples'][0][0][0][rh_indexes]
GraspStart = params['TimeGraspStart_samples'][0][0][0][rh_indexes]
GraspEnd = params['TimeGraspEnd_samples'][0][0][0][rh_indexes]
TrialEnd = params['TimeReward_samples'][0][0][0][rh_indexes]

TrialEndFull = params['TimeReward_samples'][0][0][0]

structured_data = [[[] for _ in range(3)] for _ in range(len(CueOn))] # trial x classes x channels x time
basefs = 4.8828125*10**3
discarded = set()

brain_areas = {"PMvR": (96, 96), "M1": (128, 128), "PMdR": (224, 224), "PMdL": (256, 256)}
#%%
with open('brain_areas.pkl', 'rb') as fp:
    brain_areas = pickle.load(fp)
#%%
import matplotlib.ticker as ticker
fig, ax = plt.subplots()
power, freq_=plt.psd(lfp2_data[1], Fs=freq)
plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5, alpha=1)
plt.semilogy(freq_[256:-256], power[256:-256])
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))

plt.show()
#%%
# Power Spectral Densitity plot
for x in lfp1_data:
    plt.psd(x, Fs=freq)
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
np.save('lfp1_zap_v2', lfp1_data)
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

#%%

print(brain_areas)
print("Bad Channel Rejection")
print("\tProcessing LFP1 data...")
remaining_channels, bad_channels = std_bad_channels(lfp1_data)
process_indexes(brain_areas, bad_channels)
lfp1_data = lfp1_data[remaining_channels]
print(brain_areas)
#%%
print("\tProcessing LFP2 data...")
remaining_channels, bad_channels = std_bad_channels(lfp2_data)
process_indexes(brain_areas, bad_channels, offset=128)
lfp2_data = lfp2_data[remaining_channels]
print(brain_areas)

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
# PMvR
eeg_data = lfp1_data[:brain_areas["PMvR"][1],:]
eeg_data = np.transpose(eeg_data)
# Create the info structure
n_samples, n_channels = eeg_data.shape

# Create an instance of ICA
n_components = n_channels
ica = FastICA(n_components=n_channels, random_state=42)

# Fit the ICA model to your data
eeg_data_t = ica.fit_transform(eeg_data)
#%%
np.save('PMvR', eeg_data_t)
#%%
# Plot components
fig, axs = plt.subplots(n_components, 1, figsize=[128, 6*n_components], sharey=False, sharex=True)
axs = axs.flatten()

for x in range(n_components):
    axs[x].plot(eeg_data_t[x,10000:30000], color='black', linewidth=1.5)

plt.show()
#%%
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
    return np.array(features).T
print(mara(ica, eeg_data_t, freq))
#%%
############################################################
#                       ClusterMARA                        #
############################################################

# PMvR
ica, eeg_data_t = perform_ica(lfp1_data, 0, brain_areas["PMvR"][1], random_state=42, save="")

"""
---------------------------------------------------------------
END OF PREPROCESSING
---------------------------------------------------------------
"""
#%%
# show EEGs
fig, ax = plt.subplots(facecolor='none')
plt.subplots_adjust(hspace=0.2)  # Adjust the vertical spacing between subplots

# Remove background and axes from each subplot
ax.axis('off')

# Plot the data with black lines
ax.plot(lfp1_data[0,10000:20000],color="black", linewidth=0.5)


# Adjust the figure size to make the plot more flat
fig.set_size_inches(8, 2)  # Increase the width and reduce the height as per your preference

# Show the plot
plt.show()
#%%

for i in range(len(CueOn)):
    templs = list()
    cOn = CueOn[i]
    cOff = CueOff[i]
    gStart = GraspStart[i]
    gEnd = GraspEnd[i]
    tEnd = TrialEnd[i]

    
    # prereach
    fstart = int(cOn//ratio)
    fend = int(cOff//ratio)
    duration = fend-fstart
    midpoint = (fend-fstart)//2
    fend = fstart + midpoint + 250
    fstart = fstart + midpoint - 250
    structured_data[i][0] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration < 501:
        discarded.add(i)
    # reach
    fstart = int(cOff//ratio)
    fend = int(gStart//ratio)
    duration = fend-fstart
    midpoint = (fend-fstart)//2
    fend = fstart + midpoint + 250
    fstart = fstart + midpoint - 250
    structured_data[i][1] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration < 501:
        discarded.add(i)
    # grasp
    fstart = int(gStart//ratio)
    fend = int(gEnd//ratio)
    duration = fend-fstart
    fend = fstart + midpoint + 250
    fstart = fstart + midpoint - 250
    structured_data[i][2] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration < 501:
        discarded.add(i)
    # postgrasp
    """
    fstart = int(gEnd//ratio)
    fend = int(tEnd//ratio)
    if(fstart>fend):
        print("postgrasp wtf", fstart-fend)
    duration = fend-fstart
    fend = fstart + midpoint + 250
    fstart = fstart + midpoint - 250
    structured_data[i][3] = lfp1_data[:, fstart:fend].tolist() + lfp2_data[:, fstart:fend].tolist()
    if duration < 501:
        discarded.add(i)"""

for x in sorted(list(discarded),reverse=True):
    structured_data.pop(x)

structured_data = np.array(structured_data)

np.save('structured_data', structured_data)
#%%

"""
prev = TrialEndFull[0]
sumt = 0
for i in range(1,len(TrialEndFull)):
    curr = TrialEndFull[i]
    for y in streams:
        ratio = streams[y]['ratio']
        fs = streams[y]['fs']
        sdata = streams[y]['data']
        fstart = int(prev//ratio)
        fend = int(curr//ratio)
        if y == 'LFP1':
            print("length trial", (fend-fstart)*1/float(fs))
            sumt += fend-fstart
    prev = curr
print(sumt)


for y in streams:
    templs = []
    ratio = streams[y]['ratio']
    sdata = streams[y]['data']
    print(y)
    print(sdata.shape)
    # prereach
    for start, end in zip(CueOn, CueOff):
        fstart = int(start//ratio)
        fend = int(end//ratio)
        print(fstart, fend)
        templs.append(sdata[:, fstart:fend])
    # reach
    # grasp
    # postgrasp
    print(templs[0])
    print("-"*100)
"""

# %%
from warnings import warn
from scipy.signal import iirfilter, filtfilt, welch
from tqdm import tqdm
import numpy as np
from meegkit.dss import dss_line
from sklearn.decomposition import FastICA
from sklearn.cluster import DBSCAN
from numpy.matlib import repmat
from scipy.optimize import curve_fit
from scipy.stats import skew
from math import floor

def steps_subset():
    substeps = [
            "Bandpass",
            "ZapFilter",
            "BadChannelRejection",
            "ClusterMARA",
        ]
    return substeps



class NeuroClean:

    def __init__(self, lowpass=1., highpass=500., linefreq=60.,verbose=True, random_state=None, ica_max_iters=500, ica_tol=0.0001):
        self.data = None
        self.components = None
        self.steps_performed = None
        self.lp = lowpass
        self.hp = highpass
        self.verbose = verbose
        self.linefreq = linefreq
        self.random_state = random_state
        self.ica = None
        self.max_iter = ica_max_iters
        self.tol = ica_tol

    
    def preprocess(self, data, frequency, class_info=None, substeps=None):
        self.fs = frequency
        if substeps is None:
            substeps = steps_subset()
        else:
            for s in substeps:
                if s not in steps_subset():
                    raise Exception(str(s) + " is not a step in the subsets. Please use steps_subset() to recover posible steps.")

        self.steps_performed = substeps
        with tqdm(total=len(substeps), desc="NeuroClean processing") as pbar:
            if self.data != None:
                warn("[WARNING] Calling preprocess with previously processed data replaces former data.")
            self.data = data
            if "Bandpass" in substeps:
                self.__bandpass()
                pbar.update(1)
            if "ZapFilter" in substeps:
                self.__zapfilter()
                pbar.update(1)
            if "BadChannelRejection" in substeps:
                self.__badchannelrej()
                pbar.update(1)
            if "ClusterMARA" in substeps:
                self.__ica()
                self.__clustermara()
                pbar.update(1)

    def __bandpass(self):
        b,a = iirfilter(3, [self.lp/self.fs,self.hp/self.fs], btype='bandpass', ftype='butter')
        for i, d in tqdm(enumerate(self.data), desc="Bandpassing"):
            self.data[i] = filtfilt(b, a, d)

    def __zapfilter(self):
        self.data = np.transpose(self.data, (1, 0))
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1], 1))
        self.data = self.data.astype(np.float64)
        
        self.data, _ = dss_line(self.data, fline=self.linefreq,sfreq=self.fs,blocksize=8192)

        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1]))
        self.data = np.transpose(self.data, (1, 0))
        self.data = self.data.astype(np.int16)

    def __std_bad_channels(self):
        all_channels = np.arange(self.data.shape[0])
        remaining_channels = all_channels.copy()
        k = 0  # iteration counter
        sd_pk = np.inf  # std of all individual channel std's
        std_all = np.std(self.data, axis=1)
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
        #print("\trejecting channels:", bad_channels)
        return remaining_channels, bad_channels

    def __badchannelrej(self):
        remaining_channels, bad_channels = self.__std_bad_channels()
        self.data = self.data[remaining_channels]
    
    def __ica(self):
        self.data = np.transpose(self.data)
        n_channels = self.data.shape[1]

        # Create an instance of ICA
        self.ica = FastICA(n_components=n_channels, random_state=self.random_state, max_iter=self.max_iter, tol=self.tol)

        # Fit the ICA model to your data
        self.components = self.ica.fit_transform(self.data)

    def __mara(self, ica, eeg_d, freq):
        #print("MARA...")
        patterns = ica.mixing_
        patterns = patterns/repmat(np.std(patterns,axis=0), patterns.shape[0], 1)
        features = [[0 for _ in range(patterns.shape[0])] for _ in range(6)]

        #print("\tComputing features...")
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
        icacomps = icacomps/repmat(np.std(icacomps,axis=0), icacomps.shape[0], 1)
        icacomps = icacomps.conj().T


        def myfun(xdata, *x):
            return np.exp(x[0])/np.power(xdata, np.exp(x[1])) - x[2]

        for ic in range(icacomps.shape[0]): # icacomps.shape[0]
            #print(".",end="")
            # =========================================================
            # Proc spectrum for channel
            # =========================================================
            frq, pxx = welch(icacomps[ic],window=[1]*freq_ds,noverlap=None,fs=freq_ds)
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
        #print("\nCompleted")
        return np.array(features).T

    def __clustermara(self):
        clustering = DBSCAN(eps=2, min_samples=2)
        mara_features = self.__mara(self.ica, self.components, self.fs)
        fitted = clustering.fit(mara_features)
        rejected = np.where(fitted.labels_ == -1)[0]
        self.components[:, rejected] = 0
        self.ica.inverse_transform(self.components, copy=False)
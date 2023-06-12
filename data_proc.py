import mat73
import numpy as np
import pickle

# Load LFP data from DanCause Laboratory
data = mat73.loadmat('lfp_data.mat')

streams = data["lfp_data2"]["streams"]
# This is the clock frequency of the data from the Laboratory, you may change it to the sampling frequency of the LFP
basefs = 4.8828125*10**3

# This was used to calculate the ratio of clock and sampling frequency, it is 1 if the clock is the same as the sampling rate 
for y in streams:
    if float(streams[y]['fs']) != basefs:
        streams[y]['ratio'] = float(basefs/streams[y]['fs'])
    else:
        streams[y]['ratio'] = 1

meta = dict()
meta['ratio'] = streams['LFP1']['ratio']
meta['fs'] = streams['LFP1']['fs']

# Save both LFP as two separate files
lfp1_data = streams['LFP1']['data']
lfp2_data  = streams['LFP2']['data']

np.save("lfp1_data",lfp1_data)
np.save("lfp2_data",lfp2_data)

with open('meta.pkl', 'wb') as fp:
    pickle.dump(meta, fp)
    print('dictionary saved successfully to file')

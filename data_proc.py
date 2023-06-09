#%%
import mat73
import numpy as np
import pickle
"""
events = sio.loadmat('events_2018_06_06_Y.mat')

print(events['events'].dtype.names)

for x in events['events'].T:
    print("---------------------------")
    print(x)
"""
data = mat73.loadmat('lfp_data.mat')
#%%

streams = data["lfp_data2"]["streams"]
basefs = 4.8828125*10**3

for y in streams:
    if float(streams[y]['fs']) != basefs:
        streams[y]['ratio'] = float(basefs/streams[y]['fs'])
    else:
        streams[y]['ratio'] = 1

meta = dict()
meta['ratio'] = streams['LFP1']['ratio']
meta['fs'] = streams['LFP1']['fs']

lfp1_data = streams['LFP1']['data']
lfp2_data  = streams['LFP2']['data']
#%%
np.save("lfp1_data",lfp1_data)
np.save("lfp2_data",lfp2_data)
#%%
with open('meta.pkl', 'wb') as fp:
    pickle.dump(meta, fp)
    print('dictionary saved successfully to file')
# %%

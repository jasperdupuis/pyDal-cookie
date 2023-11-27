"""

Scratch file

"""

"""

Make a percentile plot of the received levels for all freqs

requires: result dictionary object from 400_SLR_batched_with_transforms

"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

data            = result['North'] #Concatenated zero mean dB data
data            = result['South']
f               = result['Frequency']
N_f             = data.shape[0]

stds            = np.std(data,axis=1)
stds_lin        = _vars.REF_UPA * 10 ** ( ( data + 50) / 10) # add 50 for computation ease, subtract before finishing
stds_lin        = np.std(stds_lin,axis=1)
stds_from_lin   = 10*np.log10(stds_lin / _vars.REF_UPA) -50 # subtract 50 for computation ease.

plt.figure();
plt.plot(f,stds,label='Deviation calculated from dB');
plt.plot(f,stds_from_lin,label='Deviation calculated linearly then cast to dB')
plt.xscale('log')

n = 11 # change this value for the number of iterations/percentiles
colormap = cm.Blues # change this for the colormap of choice
percentiles = np.linspace(0,100,n)

SDist=np.zeros( ( N_f , n ) ) 
for i in range(n):
      SDist[:,i]=np.percentile(data,percentiles[i],axis=1)

half = int((n-1)/2)

fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
# ax1.plot(np.arange(0,N_f,1), SDist[:,half],color='k')
ax1.plot(f, SDist[:,half],color='k')
for i in range(half):
    ax1.fill_between(np.arange(0,max(f),10), SDist[:,i],SDist[:,-(i+1)],color=colormap(i/half))

ax1.set_ylim(-11,11)
ax1.plot(f,stds_from_lin,linestyle='dashed',color='r')
ax1.set_title('Percentile plot of zero-mean RL data', fontsize=10)
ax1.tick_params(labelsize=11.5)
ax1.set_xlabel('Frequency (Hz)', fontsize=10)
ax1.set_ylabel('zero-mean RL data (dB ref 1 uPa^2)', fontsize=10)
ax1.set_xscale('log')
fig.tight_layout()











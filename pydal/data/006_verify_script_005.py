# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:28:56 2023

@author: Jasper

need to verify from raw data that RL N and RL S are identical.

Get the raw time series, calculate the stft on them, and then go from there.

For an arbitrary run, this should return that the STFT results are not the same.

visualization might be nice but a numeric way to state this would also be a good thing ot have...

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import importlib
scripts_data_005 = importlib.import_module('pydal.data.005_range_files_to_h5')

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


runid = r'DRJ2PB11EX00WB'
fname_s = r'RUN_ES0451_DYN_032_000_WEST_Shyd_PORT_TM.bin'
fname_n = r'RUN_ES0451_DYN_032_000_WEST_Nhyd_STBD_TM.bin'
    
res = scripts_data_005.range_binary_to_dictionary(
    fname_s, fname_n)

south = res['South']
north = res['North']

p_T_hyd     = _vars.T_HYD
p_fs_hyd    = _vars.FS_HYD
p_window    = np.hanning(p_T_hyd * p_fs_hyd)
overlap_n   = int ( _vars.OVERLAP * p_fs_hyd )

s1 = np.sum(p_window)
s2 = np.sum(p_window**2) # completeness - not used by me. STFT applies it.

f,s_t,s_z = signal.stft(res['South'],
                      p_fs_hyd,
                      window = p_window,
                      nperseg = len(p_window),
                      noverlap = overlap_n,
                      nfft = None,
                      return_onesided = True)
f,n_t,n_z = signal.stft(res['North'],
                      p_fs_hyd,
                      window = p_window,
                      nperseg = len(p_window),
                      noverlap = overlap_n,
                      nfft = None,
                      return_onesided = True)
s_z     = 2 * (np.abs( s_z )**2) / ( s2)        # PSD
s_z     = s_z * s1                              # stft applies 1/s1, reverse this
n_z     = 2 * (np.abs( n_z )**2) / ( s2)        # PSD
n_z     = n_z * (s1)                            # stft applies 1/s1, reverse this

ss      = 10*np.log10(np.abs(s_z))
nn      = 10*np.log10(np.abs(n_z))
d       = nn - ss

ext = [ s_t[0],s_t[-1],f[0],f[-1] ]
plt.figure();plt.imshow( d , origin='lower',aspect='auto',extent=ext)
plt.figure();plt.imshow(ss,origin='lower',aspect='auto',extent=ext)
"""
From the above images conclude that 

a) the heatmap is correctly oriented (i.e. freq axis correctly labelled)
b) in logspace, stft(n) =/= stft(s)

Would not like to look at the difference under a smoothing operation (convolution)
to see if this is the source of my issue in scripts 131 & 134


In ss and nn, axis 0 is the frequency axis and axis 1 is the time axis
i.e. ss[1,:] returns a 42-long time series
"""

kern    = np.ones(11)
kern    = np.reshape(kern,(1,11))
ssc     = signal.convolve2d(ss,kern)
nnc     = signal.convolve2d(nn,kern)
dc      =  nnc - ssc # the delta of the convolved results

# shape shows time domain is convolved by extension of time axis
ssc.shape 
# visualize a single hydrophone result, looks good still.
plt.figure();plt.imshow(ssc,origin='lower',aspect='auto',extent=ext) 

""" 
below figure shows significant differences, so the error causing RL_N
and RL_S to show numerically zero error lies further in the processing
"""

plt.figure();plt.imshow(dc,origin='lower',aspect='auto',extent=ext) #visualize



"""
ERROR LIES IN FURTHER SCRIPTS!
"""

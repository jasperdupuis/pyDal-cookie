# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:01:20 2024

@author: Jasper
"""

import numpy as np
import h5py as h5

import matplotlib.pyplot as plt

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis

freq_max    = 500
freq_100_lo = 88    # for later, if want to look at an OTO Band
freq_100_hi = 111   # for later, if want to look at an OTO Band

runID_E = r'DRF2PB03AA00WB'
runID_W = r'DRF2PB03AA00EB'
directory = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\spectrograms\hdf5_spectrogram_bw_1.0_overlap_90\\'


#
#
# Eastbound run
fname_E = runID_E + r'_data_timeseries.hdf5'
fullname_E = directory +fname_E

h_E = h5.File(fullname_E,'r')
array       = h_E['North_Spectrogram'][:]
freqs       = h_E['Frequency'][:]
timesteps   = h_E['Time'][:]
ys          = h_E['Y'][:]
h_E.close()
del(h_E)

index_max           = pydal.utils.find_target_freq_index(
    p_f_targ    = freq_max, 
    p_f_basis   = freqs)

ys_no_overlap        = ys[::10]
array_no_overlap     = array[:index_max,::10]
gram = 10*np.log10(array_no_overlap/_vars.REF_UPA)

fig1,ax1 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
ax1.imshow(gram,aspect='auto',origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

figname = r'Eastbound 15kt spectrogram'
fig1.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
fig1.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
plt.show()
    
#
#
#
# Now the westbound reciprocal run    
fname_W = runID_W + r'_data_timeseries.hdf5'
fullname_W = directory +fname_W

h_W = h5.File(fullname_W,'r')
array       = h_W['North_Spectrogram'][:]
freqs       = h_W['Frequency'][:]
timesteps   = h_W['Time'][:]
ys          = h_W['Y'][:]
h_W.close()
del(h_W)

array_no_overlap     = array[:index_max,::10]
gram = 10*np.log10(array_no_overlap/_vars.REF_UPA)

fig2,ax2 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
ax2.imshow(gram,aspect='auto',origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')


figname = r'Westbound 15kt spectrogram'
fig2.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
fig2.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.png',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.show()





# -*- coding: utf-8 -*-
"""

Finally!

Merge the two different data products, synthetic and measured data from the 
spectrogram hdf5 files and the RAM TL model CSVs

Accomplish this by inserting the RAM TL data in to the existing
spectrogram hdf5 files that are output from script # 010

//WORK IS ONGOING.

//TODO : Do this generation for all runs so that other procedures can be sped up.

"""


import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


import UWAEnvTools
from UWAEnvTools import locations
from UWAEnvTools.environment import Approximations
from UWAEnvTools.singleTL.RAM import read_XY_TL_df,read_interpolate_plot_TL_df

location = 'Patricia Bay'
hyd = 'North' 
RUNID       = r'DRJ3PB09AX01EB'
FREQ        = 20
# For looping must incoroporate below somehow.
# Recall freq = 63 didn't compute TL for some reason.
# f = np.arange(10,250)
# freqs = []
# for fr in f:
#     if fr == 63 : continue
#     freqs.append(fr)
# freqs = np.array(freqs)


the_location = locations.Location(location)
dir_ram = _dirs.DIR_RAM_DATA
dir_spec = _dirs.DIR_SPECTROGRAM
dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
    _vars.FS_HYD,
    _vars.T_HYD_WINDOW * _vars.FS_HYD,
    _vars.OVERLAP
    )
dir_spec = dir_spec + dir_spec_subdir + '\\'


# RECEIVED LEVEL DATA - MEASURED
# Load the target hydrophone's run's spectrogram and x-y data
gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(RUNID, dir_spec)
gram_n          = gram_dict['North_Spectrogram']
gram_s          = gram_dict['South_Spectrogram']
t               = gram_dict['North_Spectrogram_Time']
spec_x,spec_y   = pydal.utils.rotate(gram_dict['X'], gram_dict['Y'], _vars.TRACK_ROTATION_RADS)
if len( t ) < len(spec_x): # trim the last xy to make t_n fit.
    spec_x,spec_y = spec_x[:-1],spec_y[:-1] 
   

# TRANSMISSION LOSS DATA - SYNTHETIC
# NOTE: This is frequency dependent to load!
# RAM model results for a particular frequency.
spec_f_index = pydal.utils.find_target_freq_index( # find the frequency in the spectrogram array indexing
    FREQ,
    gram_dict['Frequency'])
fname_TL = dir_ram + str(FREQ).zfill(4) + r'_' + hyd+ '.csv'
X,Y,TL = read_XY_TL_df(fname_TL)

# Visualize this frequency's TL profile over the range extent.
# read_interpolate_plot_TL_df(
#     fname_TL,
#     p_xlim = x_lim,
#     p_ylim = y_lim,
#     p_nstep = 100,
#     p_comex = 100)

TL_interped = \
    interpolate.griddata(
        (X,Y),
        TL,
        (spec_x,spec_y))


# if __name__ == '__main__':
#     x=1

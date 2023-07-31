# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:03:02 2023

@author: Jasper

The hunt for why RL_N and RL_S are the same in later work continues.

Having written 006 to check 005, now need to check the steps taken in 
010 to verify that error does not creep in there.

Start with reading in the same run data as last time where I know there was
no issue.


"""
import h5py as h5

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import importlib
scripts_data_005 = importlib.import_module('pydal.data.010_range_files_to_spec_and_xy')

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs



runID       = r'DRJ2PB11EX00WB'
p_hydro_dir = _dirs.DIR_HDF5_HYDROPHONE
p_cal_dir   = _dirs.DIR_HYDRO_CALS


p_dir       = p_hydro_dir
p_runID     = runID 
result = dict()
fname = p_dir + p_runID + r'_range_hydrophone.hdf5'
h = h5.File(fname)
for key in list(h.keys()):
    result[key] = h[key][:]
h.close()

# res = pydal.utils.get_hdf5_hydrophones_file_as_dict(runID,p_hydro_dir)

s = res['North']
n = res['South']

s == n



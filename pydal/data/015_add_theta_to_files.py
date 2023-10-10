# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:51:00 2023

@author: Jasper

compute and add theta_i and phi_i to the spectrogram datasets.

theta is defined as the angle from ship to hydrophone in the ship's reference
frame (theta is with respect to the bow, 90 is PORT, etc)

phi is defined as the angle from hydrophone to ship in the global reference frame
(phi is with respect to the POSITIVE Y axis, not standard cartesian!)
(this is to make phi closest to true north orientation)

FORWARD - PORT - UP axis orientations for theta.

"""

import numpy as np
import pandas as pd
import h5py as h5

import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal.utils

# These are FIXED in the rotated x-y system.
# Eastbound is 100 to -100 along y axis.
# Westbound is -100 to 100 along y axis.
h_n             = (  100, 0 )
h_s             = ( -100, 0 )
cpa             = ( 0 , 0 )

p_dir_spec      = _dirs.DIR_SPECTROGRAM

if __name__ == '__main__':
    DYNAMIC_ONLY    = True
    DATA_2020_ONLY  = True 
    
    local_df        = pd.read_csv(_dirs.TRIAL_MAP)
    list_run_IDs    = local_df[ local_df.columns[1] ].values
    if DYNAMIC_ONLY:
        list_run_IDs = [x for x in list_run_IDs if x[:3]=='DRJ']

    overlap = _vars.OVERLAP
    window_length = _vars.T_HYD_WINDOW
    window = np.hanning(_vars.FS_HYD * window_length)
    fs_hyd = _vars.FS_HYD

    local = pydal.utils.create_dirname_spec_xy(fs_hyd, len(window), overlap)
    data_dir = _dirs.DIR_SPECTROGRAM + local + '\\'
    

        
    # Create full path name from parameters.
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    dir_spec = p_dir_spec + dir_spec_subdir
    
    run_list = pydal.utils.get_all_runs_in_dir(dir_spec)
    # run_list_TEST = [run_list[20]]  # pick a random one.
    
    for runID in run_list:
    # for runID in run_list_TEST:
        fname_hdf5 = dir_spec + r'\\'+ runID + r'_data_timeseries.hdf5'

        temp = dict()
        spec_dict = \
            pydal.utils.load_target_spectrogram_data(
                runID, dir_spec)
        x               = spec_dict['X'] 
        y               = spec_dict['Y']
        
        R_n             = np.sqrt( ( x - h_n[0] )**2 + (y - h_n[1]) ** 2 )
        R_s             = np.sqrt( ( x - h_s[0] )**2 + (y - h_s[1]) ** 2 )
        
        CPA_n           = np.where( R_n - min( R_n ) == 0) [0] [0]
        CPA_s           = np.where( R_s - min( R_s ) == 0) [0] [0]
        
        R_CPA_n_hyd     = R_n [ CPA_n ] * -1
        R_CPA_s_hyd     = R_s [ CPA_s ]
        
        if y[0] < 0 : # Westbound: @CPA south hyd is to port(+x), north is to stbd (-x).
            R_CPA_n_ship         = R_n[ CPA_n ] * -1
            R_CPA_s_ship         = R_n[ CPA_s ]
        if y[0] > 0 : # Eastbound: @CPA south hyd is to stbd(-x), north is to stbd (+x).
            R_CPA_n_ship         = R_n[ CPA_n ]
            R_CPA_s_ship         = R_s[ CPA_s ] * -1
        
        
        
        D_n             = np.sqrt( ( x - x[CPA_n] )**2 + (y - y[CPA_n]) ** 2 )
        D_n [ CPA_n:]   = D_n[CPA_n:] * -1
        D_s             = np.sqrt( ( x - x[CPA_s] )**2 + (y - y[CPA_s]) ** 2 )
        D_s [ CPA_s:]   = D_s[CPA_s:] * -1
        
        temp['North_Theta'] = np.arctan2( R_CPA_n_ship , D_n ) * 180 / 3.14
        temp['South_Theta'] = np.arctan2( R_CPA_s_ship , D_s ) * 180 / 3.14
        
        temp['North_Phi']   = np.arctan2 ( D_n , R_CPA_n_hyd ) * 180 / 3.14
        temp['North_Phi']   = np.arctan2 ( D_s , R_CPA_s_hyd ) * 180 / 3.14
        
        with h5.File(fname_hdf5, 'a') as file:
            for data_type,data in temp.items():
                # note that not all variable types are supported but string and int are
                file[data_type] = data
    
    
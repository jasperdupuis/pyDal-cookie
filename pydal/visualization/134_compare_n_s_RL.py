# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:25:50 2023


Compare north and south hydrophones directly for a given frequency
as a function of x y

heavily based on 131.

@author: Jasper
"""


# -*- coding: utf-8 -*-
"""

For a given frequency, visualize its transmission loss as a function of 
time / R from north hydrophone. For now R is hard coded with +/- 141 on 
the entire time series, this should match the 200m track.

// TODO Work is ongoing!



"""

import numpy as np

import matplotlib.pyplot as plt

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


N_AVE           = 11 #for a smoothing window
p_freq          = 313
p_dist_to_CPA   = _vars.DIST_TO_CPA
p_location      = _vars.LOCATION
p_dir_spec      = _dirs.DIR_SPECTROGRAM
p_fs_hyd        = _vars.FS_HYD
p_n_window      = _vars.T_HYD_WINDOW * _vars.FS_HYD
p_overlap       = _vars.OVERLAP

dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
    p_fs_hyd,
    p_n_window,
    p_overlap
    )
dir_spec = p_dir_spec + dir_spec_subdir + '\\'

run_list = pydal.utils.get_all_runs_in_dir(dir_spec)
run_list = pydal.utils.get_run_selection(
    run_list,
    p_type='DR',
    p_mth='J',
    p_machine = 'X',
    p_speed = 'X',
    p_head = 'W')

gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(run_list[0], dir_spec)
f_basis           = gram_dict['Frequency']
freq_index        = pydal.utils.find_target_freq_index(p_freq, f_basis)


# Result concatenation
south   = 'South'
north   = 'North'
x       = []
y       = []
t       = []
RL_s    = []
TL_s    = []
RL_n    = []
TL_n    = []
runs    = []
count = 0
for runID in run_list:
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(runID, dir_spec)
    t.append(gram_dict[ south + '_Spectrogram_Time'])
    #south
    s = 10*np.log10(gram_dict[ south +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
    RL_s.append(s)
    #north
    n = 10*np.log10(gram_dict[ north +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
    RL_n.append(n)
    runs.append(runID)
    count+= 1
    if count > 5: break

ave_kernel = np.ones( N_AVE ) / N_AVE
fig, axs = plt.subplots(nrows = 2)
fig.suptitle('Hydrophone comparison westbound RL and TL for ' + str(p_freq) + ' Hz\n')
for run,time,rl_s,rl_n in zip(runs,t,RL_s,RL_n):
    s = rl_s - np.mean(rl_s)
    s = np.convolve( s , ave_kernel , mode='same')
    n = rl_n - np.mean(rl_n)
    n = np.convolve( n , ave_kernel , mode='same')
    Y = time - np.mean(time)
    Y = Y / np.max(Y) 
    Y = Y * 100
    if run[-2] == 'E': # Special treatment for eastbound runs 
        continue            # skip eastbound runs
        # rx = rx [::-1]    # or flip them
    axs[0].plot( Y , s, label='South' )#, label = run )
    axs[0].plot( Y , n, label='North' )#, label = run )
    axs[1].plot( Y , s - n )#, label = run )


axs[0].axhline(0,linewidth=2) # zero mean line
R = np.sqrt(Y**2 + 100 ** 2) # distance to hydrophone (nominal)
logR = 20 * np.log10(R)
zero_logR = logR - 40
axs[0].plot ( Y, -1*zero_logR,linewidth=3,label = 'Zero mean of -20logR') # zero mean 20logR model
axs[0].set_ylabel('RL (dB)')
axs[1].set_ylabel('Delta RL (dB)')
fig.supxlabel('Distance from CPA (m)')
axs[0].legend()






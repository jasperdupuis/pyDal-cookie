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
import scipy.signal as signal
import time

import matplotlib.pyplot as plt

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


N_AVE           = 9 # for a smoothing window, 9 or 11 should be comparable to 0% overlap timeseries (90% is used still)
p_freq          = 478
p_speed         = '19'
p_head          = 'X'
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
    p_speed = p_speed,
    p_head = p_head)

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

start = time.time()
delta = [] # in the same order as runs list above, has delta RL (south - north ) for the whole gram
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
    delta.append( 10 * np.log10 ( gram_dict[ south + '_Spectrogram' ] )  \
                 - 10 * np.log10 ( gram_dict[ north + '_Spectrogram'] ) )
    # if count > 5: break

ave_kernel = np.ones( N_AVE ) / N_AVE
fig, axs = plt.subplots(nrows = 2, figsize=(10,8))
if p_head =='W':
    fig.suptitle('Hydrophone comparison westbound RL and TL for ' + str(p_freq) + ' Hz\n')
if p_head =='E':
    fig.suptitle('Hydrophone comparison eastbound RL and TL for ' + str(p_freq) + ' Hz\n')

for run,times,rl_s,rl_n in zip(runs,t,RL_s,RL_n):
    s = rl_s - np.mean(rl_s)
    s = np.convolve( s , ave_kernel , mode='same')
    n = rl_n - np.mean(rl_n)
    n = np.convolve( n , ave_kernel , mode='same')
    Y = times - np.mean(times)
    Y = Y / np.max(Y) 
    Y = Y * 100
    axs[0].plot( Y , s, label='South' )#, label = run )
    axs[0].plot( Y , n, label='North' )#, label = run )
    axs[1].plot( Y , s - n )#, label = run )

end = time.time()
print('elapsed seconds : ' + str(end-start))

axs[0].axhline(0,linewidth=2) # zero mean line
R = np.sqrt(Y**2 + 100 ** 2) # distance to hydrophone (nominal)
logR = 20 * np.log10(R)
zero_logR = logR - 40
axs[0].plot ( Y, -1*zero_logR,linewidth=3,label = 'Zero mean of -20logR') # zero mean 20logR model
axs[0].set_ylabel('RL (south, dB)')
axs[1].set_ylabel('Delta RL (south - north, dB)')
fig.supxlabel('Distance from CPA (m)')
# axs[0].legend()




"""

kernel_2d = np.ones((3,3))/9
start = time.time()
z = delta[19] [ : 2000 , :]
zz = signal.convolve2d( delta[ 19 ], kernel_2d )
end = time.time()
print(str(end-start))
plt.figure();
plt.imshow(
    z,
    aspect='auto',
    origin='lower');
plt.colorbar()

"""
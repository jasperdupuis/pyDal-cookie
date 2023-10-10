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


PLOT_TL = False

N_AVE           = 11 #for a smoothing window
p_freq          = 11
# p_hydro         = _vars.HYDROPHONE
p_hydro         = 'SOUTH'
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

gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(run_list[0], dir_spec)
f_basis           = gram_dict['Frequency']
freq_index        = pydal.utils.find_target_freq_index(p_freq, f_basis)


# Result concatenation
# hydro   = p_hydro.capitalize()
hydro   = 'North'
x       = []
y       = []
t       = []
RL      = []
TL      = []
runs    = []
for runID in run_list:
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(runID, dir_spec)
    t.append(gram_dict[ hydro + '_Spectrogram_Time'])
    TL.append(gram_dict[ hydro + '_RAM_TL_interpolations'][str(p_freq).zfill(4)])
    rx = 10*np.log10(gram_dict[ hydro +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
    # RL.append(rx - np.mean(rx))
    RL.append(rx)
    runs.append(runID)
    break

# # Try looking at just one run
# index = 10
# fig, axs = plt.subplots(2)
# fig.suptitle('RL and TL for ' + str(p_freq) + ' Hz, \n Run: '+runs[index])
# run,time,loss,rx = runs[index],t[index],TL[index],RL[index]
# R = time - np.mean(time)
# R  = R / np.max(R)
# R = R * 141
# if run[-2] == 'E': #Flip eastboudn runs.
#     R = R [::-1]
# if len(time) == len (loss):
#     axs[0].plot( R , rx , label = run )
#     axs[1].plot( R , loss , label = run )
# if len(time) < len(loss):
#     axs[0].plot( R , rx , label = run )
#     axs[1].plot( R , loss[:-1] , label = run )

# # Histogram an entire frequency bin
# # db domain:
# dB_all          = np.concatenate(RL).ravel()
# m               = np.mean(dB_all )
# std             = np.std(dB_all )
# dist            = np.random.randn(100000) * std
# dist            = dist  + m
# plt.figure() ; plt.hist(dB_all ,bins=50,density=True) ; plt.hist(dist,bins=50,density=True)
# # linear domain:
# lin_all         = _vars.REF_UPA * (10 ** (dB_all / 10)) 
# m               = np.mean(lin_all )
# std             = np.std(lin_all  )
# dist_lin        = np.random.randn(100000) * std
# dist_lin        = dist_lin + m
# plt.figure() ; plt.hist(lin_all,bins=50,density=True) ; plt.hist(dist_lin,bins=50,density=True)

# N_AVE=71
ave_kernel = np.ones( N_AVE ) / N_AVE
fig, axs = plt.subplots(2)
fig.suptitle('Westbound RL and TL for ' + str(p_freq) + ' Hz\n' + hydro + ' hydrophone')
for run,time,loss,rx in zip(runs,t,TL,RL):
    rx = rx - np.mean(rx)
    rx = np.convolve( rx , ave_kernel , mode='same')
    Y = time - np.mean(time)
    Y = Y / np.max(Y) 
    Y = Y * 100
    if run[-2] == 'E': # Special treatment for eastbound runs - typical to flip or ignore them.
        # skip them
        # continue
        rx = rx [::-1]
    if len(time) == len (loss):
        axs[0].plot( Y , rx )#, label = run )
        axs[1].plot( Y , loss )#, label = run )
    if len(time) < len(loss):
        axs[0].plot( Y , rx )#, label = run )
        axs[1].plot( Y , loss[:-1] )#, label = run )
axs[0].axhline(0,linewidth=2) # zero mean line
R = np.sqrt(Y**2 + 100 ** 2) # distance to hydrophone (nominal)
logR = 20 * np.log10(R)
zero_logR = logR - 40
axs[0].plot ( Y, -1*zero_logR,linewidth=3,label = 'Zero mean of -20logR') # zero mean 20logR model
axs[0].set_ylabel('Zero-mean RL (dB)')
axs[1].set_ylabel('RAM TL (dB)')
fig.supxlabel('Distance from CPA (m)')
axs[0].legend()



# plt.plot(RL[0])
# plt.plot(RL[1])
# yy = RL[0]
# zz = RL[1]
# res = yy-zz
# plt.plot(res)


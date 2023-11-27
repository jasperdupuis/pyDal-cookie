# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:25:50 2023


Compare north and south hydrophones directly for a given frequency
as a function of x y

heavily based on script 131.

Is recycled into script 190 for figure generation of paper.

@author: Jasper
"""

    
import numpy as np
import scipy.signal as signal
import time

import matplotlib.pyplot as plt

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


N_AVE           = 11 # for a smoothing window, 9 or 11 should be comparable to 0% overlap timeseries (90% is used still)

p_freq          = 550
p_mth           = 'F'
p_speed         = 'X'
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
    p_mth=p_mth,
    p_machine = 'X',
    p_speed = p_speed,
    p_head = p_head)

gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(run_list[0], dir_spec)
f_basis           = gram_dict['Frequency']
freq_index        = pydal.utils.find_target_freq_index(p_freq, f_basis)


# Result concatenation
south   = 'South'
north   = 'North'
xs      = []
ys      = []
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
    try:
        #south
        s = 10*np.log10(gram_dict[ south +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
        #north
        n = 10*np.log10(gram_dict[ north +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
        delta.append( 10 * np.log10 ( gram_dict[ south + '_Spectrogram' ] )  \
                     - 10 * np.log10 ( gram_dict[ north + '_Spectrogram'] ) )
    except:
        continue
    if not ( len(n) == len(gram_dict['X']) ):
        continue
        # something is broken on this run
    RL_n.append(n)
    RL_s.append(s)
    runs.append(runID)
    xs.append(gram_dict['X'])
    ys.append(gram_dict['Y'])

    count+= 1
    # if count > 5: break

ave_kernel = np.ones( N_AVE ) / N_AVE
fig, axs = plt.subplots(nrows = 2, figsize=(10,8))

for run,times,x,y,rl_s,rl_n in zip(runs,t,xs,ys,RL_s,RL_n):
    # time basis for sanity checking of west v east runs
    t_loc = np.arange(len(rl_s))
    t_loc = t_loc - np.mean(t_loc) # zero centered, equal on both sides
    t_loc = t_loc / np.max(t_loc) # now scaled to -1 to 1 
    t_loc = t_loc * 100
    # x axis for giggles, normalize to middle of range though
    r = (x-100)**2 + y**2 #North hydrophone
    r = np.sqrt(r)
    x = x - np.mean(x)
    # real stuff, sorts by y position only (heading agnostic)
    s = rl_s
    s = s - np.mean(rl_s)
    s = np.convolve( s , ave_kernel , mode='same')
    n = rl_n
    n = n - np.mean(rl_n)
    n = np.convolve( n , ave_kernel , mode='same')
    # Y = times - np.mean(times)
    # Y = Y / np.max(Y) 
    # Y = Y * 100
    # NORTH
        # y coordinate, this is the good entry.
    # axs[0].plot( y , n , label='North' )#, label = run )
        # Testing value below
    axs[0].plot( t_loc , n, label='North' )#, label = run )
    # axs[0].plot( r , n, label='North' )#, label = run )
    # axs[0].scatter( x , n, marker = '.', s=1, label='North' )#, label = run )
        # END TESTING VALUES
    # SOUTH
        # y coordinate, this is the good entry.
    # axs[1].plot( y , s , label='South' )#, label = run )
        # Testing value below
    axs[1].plot( t_loc , s, label='South' )#, label = run )
    # axs[1].plot( r , s, label='South' )#, label = run )
    # axs[1].scatter( x , s, marker = '.', s=1,label='South' )#, label = run )
        # DELTA
    # axs[1].plot( y , s - n )#, label = run )
        # END TESTING VALUES

end = time.time()
print('elapsed seconds : ' + str(end-start))

axs[0].axhline(0,linewidth=2) # zero mean line
R = np.sqrt(y**2 + 100 ** 2) # distance to hydrophone (nominal)
logR = 20 * np.log10(R)
zero_logR = logR - 40
# axs[0].plot ( y, -1*zero_logR,linewidth=3,label = 'Zero mean of -20logR') # zero mean 20logR model
fig.suptitle('Zero-mean spectral time series for north and south hydrophone at ' + str(p_freq) + ' Hz\n')
axs[0].set_ylabel(r'North hydrophone (dB ref ${\mu}Pa^2 m^2$)')
axs[1].set_ylabel(r'South hydrophone (dB ref ${\mu}Pa^2 m^2$)')
# axs[1].set_ylabel('Delta RL (south - north, dB)')
# fig.supxlabel('Distance from CPA (m)')
fig.supxlabel('Y-position in range X-Y system (m)')
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
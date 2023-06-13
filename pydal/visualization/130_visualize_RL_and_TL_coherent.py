"""

For a single freq, hydrophone, and run selection will plot TL and RL with their mean 
values subtracted out.

"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate,signal


import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


import UWAEnvTools
from UWAEnvTools import locations
from UWAEnvTools.environment import Approximations
from UWAEnvTools.singleTL.RAM import read_XY_TL_df,read_interpolate_plot_TL_df



def correlate_TL_RL_for_one_run_over_freqs(
        p_runID,
        p_freqs = np.arange(11,50),
        p_location = _vars.LOCATION,
        p_hydro = _vars.HYDROPHONE):
    """
    Compute the correlations over the range of provided freqs in a given run.
    """
    the_location = locations.Location(p_location)
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
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(p_runID, dir_spec)
    gram_n          = gram_dict['North_Spectrogram']
    gram_s          = gram_dict['South_Spectrogram']
    t               = gram_dict['North_Spectrogram_Time']
    spec_x,spec_y   = pydal.utils.rotate(gram_dict['X'], gram_dict['Y'], _vars.TRACK_ROTATION_RADS)
    if len( t ) < len(spec_x): # trim the last xy to make t_n fit.
        spec_x,spec_y = spec_x[:-1],spec_y[:-1] 
    if p_hydro == 'North':
        gram = gram_n
    if p_hydro == 'South':
        gram = gram_s
    
    
    
    # TRANSMISSION LOSS DATA - SYNTHETIC
    # NOTE: This is frequency dependent to load!
    # RAM model results for a particular frequency.
    result_corr = dict()
    result_cov = dict()
    for freq in p_freqs:
        spec_f_index = pydal.utils.find_target_freq_index( # find the frequency in the spectrogram array indexing
            freq,
            gram_dict['Frequency'])
        fname_TL = dir_ram + str(freq).zfill(4) + r'_' + p_hydro + '.csv'
        X,Y,TL = read_XY_TL_df(fname_TL)
        
        TL_interped = \
            interpolate.griddata(
                (X,Y),
                TL,
                (spec_x,spec_y))
            
        result_cov[str(freq).zfill(4)] = np.cov(gram[spec_f_index,:],TL_interped)
        result_corr[str(freq).zfill(4)] = np.corrcoef(gram[spec_f_index,:],TL_interped)
    return result_cov,result_corr


def load_and_display_run_freq_TLRL(
        p_runID,
        p_freq,
        p_RL_smoothing_function, #smoothing function of RL
        p_location = 'Patricia Bay',
        p_hydro = 'NORTH',
        **kwargs):
    
    the_location = locations.Location(p_location)
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
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(p_runID, dir_spec)
    gram_n          = gram_dict['North_Spectrogram']
    gram_s          = gram_dict['South_Spectrogram']
    t               = gram_dict['North_Spectrogram_Time']
    spec_x,spec_y   = pydal.utils.rotate(gram_dict['X'], gram_dict['Y'], _vars.TRACK_ROTATION_RADS)
    if len( t ) < len(spec_x): # trim the last xy to make t_n fit.
        spec_x,spec_y = spec_x[:-1],spec_y[:-1] 
    if p_hydro == 'NORTH':
        gram = gram_n
    if p_hydro == 'SOUTH':
        gram = gram_s
    spec_f_index = pydal.utils.find_target_freq_index( # find the frequency in the spectrogram array indexing
        p_freq,
        gram_dict['Frequency'])

    
    # TRANSMISSION LOSS DATA - SYNTHETIC
    # NOTE: This is frequency dependent to load!
    # RAM model results for a particular frequency.
    fname_TL = dir_ram + str(p_freq).zfill(4) + r'_' + p_hydro + '.csv'
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
    
    RL = gram_n[ spec_f_index , :]
    RL = np.convolve(p_RL_smoothing_function,RL,mode='same')
    RL_db = 10 * np.log10( RL )
    
    RL_n_f_at_0dc = RL_db
    TL_n_f_at_0dc = TL_interped

    if kwargs['DC'] :
        RL_n_f_at_0dc = RL_db - np.mean(RL_db)
        TL_n_f_at_0dc = TL_interped - np.mean ( TL_interped )
    
    fig,ax = plt.subplots( figsize = ( 8 , 5 ) )
    ax.plot( t , RL_n_f_at_0dc , label = 'RL')
    ax.plot( t , TL_n_f_at_0dc , label = 'TL')
    plt.legend()
    fig.suptitle('Run ' + RUNID + '\n TL and RL comparison \n frequency = ' + str(p_freq) + ' Hz')
    
    return fig,ax


if __name__ == '__main__':
    RUNID       = r'DRJ3PB19AX01EB'
    FREQ        = 227
    N_KERNEL = 15
    HYDRO = _vars.HYDROPHONE
    LOCATION = _vars.LOCATION
    # For looping must incoroporate below somehow.
    # Recall freq = 63 didn't compute TL for some reason.
    # f = np.arange(10,250)
    # freqs = []
    # for fr in f:
    #     if fr == 63 : continue
    #     freqs.append(fr)
    # freqs = np.array(freqs)

    
    # Demonstrate a single freq, single run overlay plot with DC bias removed
    RL_smoothing = np.ones( N_KERNEL ) / N_KERNEL
    fig,ax = load_and_display_run_freq_TLRL(
        p_runID = RUNID,
        p_freq = FREQ,
        p_RL_smoothing_function = RL_smoothing,
        p_location = LOCATION,
        p_hydro = HYDRO,
        DC = False)
    # fig.savefig('test.png')


    # Demonsrtate correlatio  over all frequencies of itnerest for a single run
    fs = np.arange(11,250)
    freqs = []
    for f in fs:
        if f == 63: continue # Freq didnt work in TL generation
        freqs.append(f)
    freqs = np.array(freqs)
    # cov_dict,corr_dict = correlate_TL_RL_for_one_run_over_freqs(
    #     p_runID = RUNID,
    #     p_freqs = freqs,
    #     p_location = LOCATION,
    #     p_hydro = HYD)

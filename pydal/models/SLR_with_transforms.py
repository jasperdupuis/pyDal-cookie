# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:39:02 2023

@author: Jasper

Explore theta as an explanatory variable. Has been adapted to work on any x,y,theta regressor
by using passed x or y transforms.

First implemented in loop, moved to vectorized 20230911.


    In a fully transformed cartesian x-y system:
    North hydrophone is at (100,0)
    South hydrophone is at (-100,0)
    Eastbound goes from (0,100) to (0,-100)
    Westbound goes from (0,-100) to (0,100)

"""

import h5py as h5
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time

import pydal.utils
import pydal.data_transforms
import pydal._directories_and_files as _dirs
import pydal._variables as _vars


MAX_FREQ_INDEX = 50000 #randomly chosen number, dont want high freq stuff
DEG_TO_RAD = np.pi / 180
STANDARD        = r'STANAG' # uses +/- 45 degrees (i.e. no trim needed)
# STANDARD        = r'ISO'# uses +/- 30 degrees
# STANDARD        = r'HEGGERNES'# Uses +/- one shiplength
CHECK_VECTORIZE = False



def array_linear_regression(regressor,p_array):
    """
    Efficient memory handling (for python) of SLR regression over
    a 2-D array of y data.
    """
    M = p_array.shape[1]
    if not (len(regressor) == M): 
        print('Error in array_linear_regression function, lengths not the same \n ')
        return
    N = p_array.shape[0]
    m , b , r , p , m_err \
        = np.zeros(N) ,np.zeros(N) ,np.zeros(N) ,np.zeros(N) ,np.zeros(N) 
    for index in range(N):
        reg = stats.linregress(regressor,p_array[index,:])
        m[index]        = reg.slope
        b[index]        = reg.intercept
        r[index]        = reg.rvalue
        p[index]        = reg.pvalue
        m_err[index]    = reg.stderr
        
    return m,b,r,p,m_err


def set_range_standard(
        p_x,
        p_y,
        p_gram,
        p_theta,
        p_standard = r'STANAG'):
    
    if p_standard == r'STANAG':
        print('STANAG method: no data trim required')
        #nothing to be done    

    if p_standard == r'HEGGERNES':
        # This uses +/- shiplength from CPA.
        # The middle third is 60 < length < 70, so good enough here for ORCA.
        m1              = len(p_x) // 3 # start middle third
        m3              = m1 * 2      # end middle third
        p_x             = p_x [ m1 : m3 ]
        p_y             = p_y [ m1 : m3 ]
        p_gram          = p_gram [ : , m1 : m3 ]
        p_theta         = p_theta [ m1 : m3]    

    if p_standard == r'ISO': 
        #should be 57m, ~50m close enough for my work for now.
        m1              = len(p_x) // 4 # start second quarter
        m3              = m1 * 3      # end third quarter 
        p_x               = p_x [ m1 : m3 ]
        p_y               = p_y [ m1 : m3 ]
        p_gram          = p_gram [ : , m1 : m3 ]
        p_theta         = p_theta [ m1 : m3]    

    return p_x,p_y,p_gram,p_theta



def SLR_with_y_transform(
        p_x,
        p_y,
        p_theta,
        p_gram,
        p_x_transform,
        p_y_transform):
    """
    For a given hydrophone determined outside this function:
    
    Pass the dependent variables, 2-D data array (spectrogram) 
    and functions with which to transform the x and y data if wanted.
    
    p_x        - regressor var x_i
    p_gram     - target var y_i
    p_function - transform for y_i
    
    returns the results in a dictionary.
    """
    regressor = p_x_transform(p_x,p_y,p_theta)
    z_arr    = p_y_transform(p_gram)
    m,b,r,p,err         = array_linear_regression(regressor, z_arr)
    result              = dict()
    result['m']         = m
    result['b']         = b
    result['r']         = r
    result['p']         = p
    result['err']         = err
    return result


def compute_regressor_variables(
        x,y):
    """
    In a fully transformed cartesian x-y system:
    North hydrophone is at (100,0)
    South hydrophone is at (-100,0)
    Eastbound goes from (0,100) to (0,-100)
    Westbound goes from (0,-100) to (0,100)
    """
    n_r     = np.sqrt(( ( x - 100 ) ** 2 ) + y**2)
    s_r     = np.sqrt(( ( x + 100 ) ** 2 ) + y**2)
    n_logR  = 20*np.log10(n_r)
    s_logR  = 20*np.log10(s_r)
    return n_r, s_r, n_logR, s_logR


def SLR_single_run_with_var_transforms(
        p_runID,
        p_dir_spec,
        p_standard = STANDARD,
        p_x_transform = pydal.data_transforms.x_transform_theta_only,
        p_y_transform = pydal.data_transforms.y_transform_0_mean_max_norm_arcsin,
        p_max_freq_ind  = MAX_FREQ_INDEX
        ):
    """
    It does what it says on the tin.
    
    p_t_start_end_percentages relates always to a //10 division of the xx and
    yy series length, to trim the data along time axis.
    e.g. (1,9) is //10 and (//10) * 9
    """
    # Get the data, trim frequency axis where appropriate.
    spec_dict = \
        pydal.utils.load_target_spectrogram_data(
            p_runID, p_dir_spec)
        
    xx, yy , n_theta, s_theta = \
        spec_dict['X'],spec_dict['Y'],\
        spec_dict['North_Theta'],spec_dict['South_Theta']
    f           = spec_dict ['Frequency'] [ : p_max_freq_ind ]
    n_gram      = spec_dict['North_Spectrogram'][ : p_max_freq_ind , :]
    s_gram      = spec_dict['South_Spectrogram'][ : p_max_freq_ind , :]
    
    
    
    if len(n_theta) > n_gram.shape[1] : 
        n_theta         = n_theta[:-1]
        s_theta         = s_theta[:-1]

    # Trim the time axis where appropriate, per STANDARD.
    x,y,n_gram_lin,n_theta          = set_range_standard(xx, yy, n_gram, n_theta,
                                                     p_standard = STANDARD)
    _,_,s_gram_lin,s_theta          = set_range_standard(xx, yy, s_gram, s_theta,
                                                     p_standard = STANDARD)
    n_gram_db,s_gram_db             = 10*np.log10(n_gram_lin), \
                                        10*np.log10(s_gram_lin)

    """
    The null hypothesis is slope = 0 when using theta to predict y
    (Run for both linear and db)
    Small p ==> reject null hypothesis.
    """
    n_result_lin    = SLR_with_y_transform(x , y , n_theta, n_gram_lin , p_x_transform, p_y_transform)
    n_result_db     = SLR_with_y_transform(x , y , n_theta, n_gram_db , p_x_transform, p_y_transform)
    s_result_lin    = SLR_with_y_transform(x , y , s_theta, s_gram_lin ,p_x_transform,  p_y_transform )
    s_result_db     = SLR_with_y_transform(x , y , s_theta, s_gram_db , p_x_transform, p_y_transform)

    result = { 'North_Linear' : n_result_lin,
              'North_Decibel'  : n_result_db,
              'South_Linear'  : s_result_lin,
              'South_Decibel'  : s_result_db,
              'Frequency'       : f}

    return result


def get_indices_according_to_standard(
        p_n,
        p_standard=STANDARD):
    """
    STANAG, ISO, HEGGERNES have different modulo math.
    """
    if p_standard == r'STANAG':
        # print('STANAG method: no data trim required')
        #nothing to be done    
        return (0,p_n)

    if p_standard == r'HEGGERNES':
        # This uses +/- shiplength from CPA.
        # The middle third is 60 < length < 70, so good enough here for ORCA.
        m1              = p_n // 3 # start middle third
        m3              = m1 * 2      # end middle third
        return (m1,m3)

    if p_standard == r'ISO': 
        #should be CPA +- 57m, ~50m close enough for my work.
        m1              = p_n // 4 # start second quarter
        m3              = m1 * 3      # end third quarter 
        return (m1,m3)
    
    if p_standard == r'50m':
        m1 = ( p_n // 8 )   # start of 2nd 8th
        m3 = m1 * 5         # end of 5th 8th
        m1 = m1 * 3         # start of 4th 8th.
        return (m1,m3)


def get_mask_array_according_to_standard(
        p_run_lengths,
        p_standard = STANDARD):
    N = 0
    for n in p_run_lengths:
        N += n
    mask = np.zeros( N )
    current_run_start_index = 0
    for n in p_run_lengths:
        start,end = get_indices_according_to_standard(n,p_standard)
        start += current_run_start_index + start
        end += current_run_start_index + end
        mask[start:end] = True
        current_run_start_index += n      
    mask=np.array(mask,dtype=bool) # convert from 1 and 0 to booleans
    return mask


def mask_data(p_dict,p_standard):

    nanmask_n   = np.isfinite(np.sum(p_dict['North'],axis=0))
    nanmask_s   = np.isfinite(np.sum(p_dict['South'],axis=0))
    nanmask     = np.logical_or(nanmask_n,nanmask_s)
    # nanmask is all the values that are NOT nan or inf, i.e. i want those.

    # now the stanag / iso / heggernes criteria:
    mask        = get_mask_array_according_to_standard(
        p_run_lengths = p_dict['Run_Lengths'], 
        p_standard = STANDARD)
    mask        = np.array(mask,dtype=bool)
    #this mask is allthe time steps that meet the track criteria, i.e. i want those

    #   combine the two masks. Must meet BOTH criteria ==> use AND
    mask = np.logical_and(nanmask,mask)

    res_dict               = dict()
    res_dict ['North']     = p_dict['North'][:,mask]
    res_dict ['South']     = p_dict['South'][:,mask]
    res_dict ['X']         = p_dict['X'][mask]
    res_dict ['Y']         = p_dict['Y'][mask]
    res_dict ['Frequency'] = p_dict['Frequency'][:_vars.INDEX_FREQ_MAX_PROCESSING]

    return res_dict


def compare_SL_nominal_vs_RL_slope_implied(
        p_ax,
        p_label, 
        p_f_values,
        p_m_values,
        p_track_dist_m  = 200,
        p_track_step    = 0.5,
        p_sl_nom        = 160,
        p_color         = 'black',
        p_linestyle     ='--'
        ):
    """
    Calculate the difference between a "true" source level p_sl_nom,
    and what that same SL subjected to the slopes observed in the batched
    SLR approach.
    """
    track_steps     = np.arange(p_track_dist_m / p_track_step) 
    track_steps     -= (len(track_steps) // 2 )
    track_steps     *= p_track_step
    track_steps     = np.reshape(track_steps,( len ( track_steps) , 1 ) )
    slope_per_100m  = np.reshape(p_m_values, ( 1 , len ( p_m_values)))

    tl_var          = np.multiply(track_steps,slope_per_100m)

    # now, create what the RL would be while accounting for the 
    # linear TL variation model.
    rl              = p_sl_nom + tl_var
    rl_lin          = _vars.REF_UPA * (10 ** ( ( rl / 10 )))
    rl_lin_mean     = np.mean(rl_lin,axis=0)
    rl_db_mean      = 10*np.log10(rl_lin_mean / _vars.REF_UPA)
    
    delta       = rl_db_mean - p_sl_nom
    p_ax.plot( p_f_values , 
              delta , 
              color = p_color,
              label=p_label, 
              linestyle = p_linestyle,
              linewidth=0.7) ; 
    
    return p_ax, p_sl_nom,rl_db_mean


def load_concat_arrays(
    p_fname         =  'concatenated_data.pkl'
    ):
    dir_spec , _    =  pydal.utils.get_fully_qual_spec_path()
    result          = pydal.utils.load_pickle_file(
        dir_spec,
        p_fname)
    print(result['Runs'])
    return result


if __name__ == '__main__':
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    p_dir_spec      = _dirs.DIR_SPECTROGRAM + dir_spec_subdir + '\\'
    run_list        = pydal.utils.get_all_runs_in_dir(p_dir_spec)
    p_head          = 'W'
    p_speed         = 'X'
    run_list        = pydal.utils.get_run_selection(
        run_list,
        p_type='DR',
        p_mth='J',
        p_machine = 'X',
        p_speed = p_speed,
        p_head = p_head)    
    
    """
    The null hypothesis is slope = 0 when using theta to predict y
    (Run for both linear and db)
    Small p ==> reject null hypothesis.
    """
    start = time.time()
    results = dict()
    for r in run_list:
        run_result = SLR_single_run_with_var_transforms(
            r,
            p_dir_spec,
            p_standard = STANDARD,
            p_x_transform = pydal.data_transforms.x_transform_y_only,
            p_y_transform = pydal.data_transforms.y_transform_0_mean,
            p_max_freq_ind  = MAX_FREQ_INDEX
            )    
        results[r] = run_result
    end = time.time()
    print('time elapsed:\t' + str(end-start) )
    
    run_result.keys()
    
    # plt.figure(); plt.plot(run_result['Frequency'],run_result['North_Linear']['p']);plt.xscale('log');plt.title('Linear arcsin p-value \nNorth, ' + STANDARD + ', ' + runID )
    # plt.figure(); plt.plot(run_result['Frequency'],run_result['North_Decibel']['p']);plt.xscale('log');plt.title('Decibel arcsin p-value \nNorth, ' + STANDARD + ', ' + runID )
    
    n_p_db  = np.zeros(MAX_FREQ_INDEX)
    n_p_lin = np.zeros(MAX_FREQ_INDEX)
    s_p_db  = np.zeros(MAX_FREQ_INDEX)
    s_p_lin = np.zeros(MAX_FREQ_INDEX)
    x = []
    for key,value in results.items():
        s_p_db      += value['South_Decibel']['m'] 
        s_p_lin     += value['South_Linear']['m']
        n_p_db      += value['North_Decibel']['m'] 
        n_p_lin     += value['North_Linear']['m']
        # x.append (value['North_Linear']['p'])
    # Average the results based on number of entries:
    n_p_db      = n_p_db / len(results.keys())
    n_p_lin     = n_p_lin / len(results.keys())
    s_p_db      = s_p_db / len(results.keys())
    s_p_lin     = s_p_lin / len(results.keys())

    # South hydrophone    
    #linear
    # plt.figure(); plt.plot(run_result['Frequency'],s_p_lin);plt.xscale('log');plt.title('Linear m-value for H_0 zero slope \nSouth, ' + STANDARD + ', averaged'  )
    # decibel
    plt.figure(); plt.plot(run_result['Frequency'],s_p_db);plt.xscale('log');plt.title('Decibel m-value for H_0 zero slope \nSouth, ' + STANDARD + ', averaged' )

    # North hydrophone
    #linear
    # plt.figure(); plt.plot(run_result['Frequency'],n_p_lin);plt.xscale('log');plt.title('Linear m-value for H_0 zero slope \nNorth, ' + STANDARD + ', averaged'  )
    # decibel
    plt.figure(); plt.plot(run_result['Frequency'],n_p_db);plt.xscale('log');plt.title('Decibel m-value for H_0 zero slope \nNorth, ' + STANDARD + ', averaged' )

    
    
    # A recipe for better sine model to estimate params for from the example at :
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # popt, pcov = optimize.curve_fit(sine_hypothesis_model,n_theta,z)
    # popt
    # pcov
    
    
    # For future debugging needs only, will use vectorized going forward.
    # There are code errors below, variable names.
    if CHECK_VECTORIZE: 
        freqindex = 71 # the particular freq bin to investigate to verify vectorization operations
        # f[71] # f[71] == 73.0 Hz, has interesting features shown at committee.
        index = freqindex
        
        spec_dict   = \
            pydal.utils.load_target_spectrogram_data(
                run_list[0], p_dir_spec)
        n_gram      = spec_dict [ 'North_Spectrogram' ]
        n_gram_lin  = n_gram
        n_gram_db   = 10*np.log10(n_gram)
        n_theta     = spec_dict [ 'North_Theta' ]
            
        # Do the linear work first:
        n_spectral_time_series      = n_gram[index,:]
        z_db                        = 10 * np.log10( n_spectral_time_series ) 
        
        z_lin           = 10**(z_db/10)
        z_lin_0         = z_lin - np.mean(z_lin)
        Dz_lin          = np.max(np.abs(z_lin_0))
        z_lin_norm      = z_lin_0 / Dz_lin #normalized on -1,1
        z_lin_arcsin    = np.arcsin(z_lin_norm)
    
        # Compute a decibel-based result too:    
        z_db_0          = z_db - np.mean(z_db) 
        Dz_db           = np.max(np.abs(z_db_0))
        z_db_norm      = z_db_0 / Dz_db #normalized on -1,1
        z_db_arcsin    = np.arcsin(z_db_norm)
        
        """
        Plot the spectral time series in time domain and dB domains
        Calculated from non-vectorized operations.
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('FPU angle (rad)')
        ax1.set_ylabel('red: decibel power')
        ax1.plot(n_theta,z_db,color='red')
        ax2.set_ylabel('blue: linear power')
        ax2.plot(n_theta,z_lin,color='blue')
        
        """
        Plot the dB and Linear arcsin timeseries, calculated using 
        non-vectorization operations
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('FPU angle (rad)')
        ax1.set_ylabel('red: arcsin of decibel power (zero mean)')
        ax1.plot(n_theta,z_db_arcsin,color='red')
        ax2.set_ylabel('blue: arcsin of linear power (zero mean)')
        ax2.plot(n_theta,z_lin_arcsin,color='blue')
    
        
        """
        Select and plot the arcsin timeseries, calculated using
        vectorization.
        """    
        z_db_arcsin_arr_sel   = pydal.data_transforms.y_transform_0_mean_max_norm_arcsin(n_gram_db)[index,:]
        z_lin_arcsin_arr_sel  = pydal.data_transforms.y_transform_0_mean_max_norm_arcsin(n_gram_lin)[index,:]
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('FPU angle (rad)')
        ax1.set_ylabel('red: arcsin of decibel power (zero mean)')
        ax1.plot(n_theta,z_db_arcsin_arr_sel,color='red')
        ax1.set_ylabel('blue: arcsin of linear power (zero mean)')
        ax2.plot(n_theta,z_lin_arcsin_arr_sel,color='blue')

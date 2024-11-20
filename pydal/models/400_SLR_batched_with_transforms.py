# -*- coding: utf-8 -*-
"""

This file treats SLR across entire sets of runs at once, rather than per
each run.

"""

import time # timing of operations


import numpy as np
import matplotlib.pyplot as plt

import pydal.utils
import pydal.data_transforms
import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal._thesis_constants as _thesis

import SLR_with_transforms


GENERATE_DATA   = False
LOAD_DATA       = False
SINGLE_YEAR     = False
COMPARE_YEARS   = False
COMPARE_P_AND_M = False
COMPARE_P_AND_M_SUPPRESS_ID_INFO = True


STANDARD        = 'STANAG'
# STANDARD        = 'ISO'
# STANDARD        = '50m'

MTH             = _vars.MTH
# MTH             = 'J'
# MTH             = 'F'
HYDRO           = 'North'

def get_and_concatenate_lists_of_grams_and_xy(
        p_run_list,
        p_dir_spec,
        p_max_freq_index = _vars.INDEX_FREQ_MAX_PROCESSING
        ):
    """
    Instead of performing SLR over each run and then trying to average,
    this function instead treats all (passed) runs at once.
    
    """
    # loop to make the first entry in data in case [0]th entry is fucked up
    # see length criteria about 15 lines down with continue condition.
    count = 0 # track where to start the exhaustive loop.
    for r in p_run_list[1:]:
        spec_dict,N   = \
            pydal.utils.get_spectrogram_file_as_dict(
                r, 
                p_dir_spec                
                )
        xx          = spec_dict [ 'X' ] 
        yy          = spec_dict [ 'Y' ] 
        # # decibels
        s           = \
            pydal.data_transforms.y_transform_0_mean( 
                10 * np.log10(
                    spec_dict [ 'South_Spectrogram' ]))[:p_max_freq_index,:]
        n           = \
            pydal.data_transforms.y_transform_0_mean( 
                10 * np.log10(
                    spec_dict [ 'North_Spectrogram' ]))[:p_max_freq_index,:]
        count += 1
        if len(xx) == len (n) :
            continue
            # something broken with this run, continue to get a good one.
        index = pydal.utils.get_max_index(len(xx),len(yy),s.shape,n.shape)
        
        x           = xx [ : index ]
        y           = yy [ : index ]
        s_concat    = s [ : , : index ]
        n_concat    = n [ : , : index ]
        run_lengths = [ index ]
        break

    for r in p_run_list[count:]:
        spec_dict,N   = \
            pydal.utils.get_spectrogram_file_as_dict(
                r, p_dir_spec)
        xx          = spec_dict [ 'X' ] 
        yy          = spec_dict [ 'Y' ] 
        # # decibels
        s           = \
            pydal.data_transforms.y_transform_0_mean( 
                10 * np.log10(
                    spec_dict [ 'South_Spectrogram' ]))[:p_max_freq_index,:]
        n           = \
            pydal.data_transforms.y_transform_0_mean( 
                10 * np.log10(
                    spec_dict [ 'North_Spectrogram' ]))[:p_max_freq_index,:]
        if len(xx) == len (n) :
            continue
            # something brokenwith this run, continue.
        index = pydal.utils.get_max_index(len(xx),len(yy),s.shape,n.shape)
        
        x           = np.concatenate((x, xx [ : index ]) )
        y           = np.concatenate((y, yy [ : index ]) )
        run_lengths.append( index )
        s_concat    = np.concatenate ((s_concat,s[ : , : index ]),axis=1)
        n_concat    = np.concatenate ((n_concat,n[ : , : index ]),axis=1) 

        # print(r)
        # break

        
    result = dict()
    result['X'] = x
    result['Y'] = y
    result['North'] = n_concat
    result['South'] = s_concat
    result['Run_Lengths'] = run_lengths
    result['Runs'] = p_run_list
    result['Transform'] = 'Zero-mean in decibel domain, original dB value unretrievable from here'
    result['Frequency'] = spec_dict['Frequency']
    return result


def store_concat_arrays(
        p_concat_dict,
        p_fname = 'concatenated_data.pkl' ):
    """
    Void passed for directory specificaitons, should come from _vars and _dirs
    """
    dir_spec ,_    =  pydal.utils.get_fully_qual_spec_path()
    pydal.utils.dump_pickle_file(
        p_concat_dict,
        dir_spec,
        p_fname)
    return



if __name__ == '__main__':    
    if MTH == 'J':
        fname = 'concatenated_data_2019.pkl'
    if MTH == 'F':
        fname = 'concatenated_data_2020.pkl'
    
    if STANDARD == 'STANAG':
        track_dist = 200;
    if STANDARD == 'ISO':
        track_dist = 114;
    if STANDARD == '50m':
        track_dist = 50;
    
    
    if GENERATE_DATA :  
        p_dir_spec,run_list = pydal.utils.get_fully_qual_spec_path()
        p_head          = 'X'
        p_speed         = 'X'
        p_beam          = 'B'
        run_list        = pydal.utils.get_run_selection(
            run_list,
            p_type='DR',
            p_mth=MTH,
            p_machine = 'X',
            p_speed = p_speed,
            p_head = p_head,
            p_beam = p_beam)    
        
        start = time.time()
        result = get_and_concatenate_lists_of_grams_and_xy(
            run_list,
            p_dir_spec,
            p_max_freq_index = _vars.INDEX_FREQ_MAX_PROCESSING
            )    
        
        store_concat_arrays(result,fname)
        end= time.time()
        print(str(end-start))

    if LOAD_DATA:
        result = SLR_with_transforms.load_concat_arrays(fname)


    if SINGLE_YEAR:
        
        result = SLR_with_transforms.mask_data(result,STANDARD)

        f       = result['Frequency']
    
        # m,b,r,p,err
        n_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = result['X'],
            p_y             = result['Y'],
            p_theta         = np.zeros_like(result['X']), #not used placeholder
            p_gram          = result['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        s_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = result['X'],
            p_y             = result['Y'],
            p_theta         = np.zeros_like(result['X']), #not used placeholder
            p_gram          = result['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
    
        if HYDRO =='North':
            plt.figure(); plt.plot(f,n_result_db['m']);plt.xscale('log');plt.title('dB / m slope for SLR \n North' +', ' + STANDARD + ', aggregated runs' )
            # sl_nom ,rl_db_n = \
            #     SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            #         p_f_values      = f,
            #         p_m_values      = n_result_db['m'],
            #         p_plot          = True,
            #         p_track_dist_m  = track_dist)
    
            fig1,ax1 = plt.subplots(1,1,figsize=(10,8))    
            ax1, sl_nom ,rl_db_n2019 = \
                SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                    p_ax = ax1,
                    p_label = '2019',
                    p_f_values      = f,
                    p_m_values      = n_result_db['m'],
                    p_track_dist_m  = track_dist,
                    p_color         = 'black',
                    p_linestyle     ='--')
    
    
    
        if HYDRO =='South':
            plt.figure(); plt.plot(f,s_result_db['m']);plt.xscale('log');plt.title('dB / m slope for SLR \n South' +', ' + STANDARD + ', aggregated runs' )
            sl_nom ,rl_db_s = \
                SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                    p_f_values      = f,
                    p_m_values      = s_result_db['m'],
                    p_plot          = True,
                    p_track_dist_m  = track_dist)
                
        
                
    if COMPARE_YEARS:
        r2019   = SLR_with_transforms.load_concat_arrays('concatenated_data_2019.pkl')
        r2020   = SLR_with_transforms.load_concat_arrays('concatenated_data_2020.pkl')
        
        r2019   = SLR_with_transforms.mask_data(r2019,STANDARD)
        r2020   = SLR_with_transforms.mask_data(r2020,STANDARD)
        
        f       = r2019['Frequency']  

        n2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        s2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )

        n2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        s2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
     
        plt.figure(); plt.plot(f,n2019_result_db['m'],label='2019');plt.xscale('log');plt.title('dB / m slope for SLR \n North' +', ' + STANDARD + ', aggregated runs' )
        plt.figure(); plt.plot(f,n2020_result_db['m'],label='2020');plt.xscale('log');plt.title('dB / m slope for SLR \n North' +', ' + STANDARD + ', aggregated runs' )
        fig1,ax1 = plt.subplots(1,1,figsize=(10,8))    
        ax1, sl_nom ,rl_db_n2019 = \
            SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                p_ax = ax1,
                p_label = '2019',
                p_f_values      = f,
                p_m_values      = n2019_result_db['m'],
                p_track_dist_m  = track_dist,
                p_color         = 'black',
                p_linestyle     ='--')
        ax1, sl_nom ,rl_db_n2020 = \
            SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                p_ax = ax1,
                p_label = '2020',
                p_f_values      = f,
                p_m_values      = n2020_result_db['m'],
                p_track_dist_m  = track_dist,
                p_color         = 'red',
                p_linestyle     = '-')
        fig1.suptitle(r'$\varepsilon$, assuming CPA reference and '+STANDARD+' standard\n North hydrophone'); plt.xscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Estimated error, $dB ref 1{\mu}Pa^2 / Hz @ 1m$')
        ax1.legend()
    
        plt.figure(); plt.plot(f,s2019_result_db['m'],label='2019');plt.xscale('log');plt.title('dB / m slope for SLR \n South' +', ' + STANDARD + ', aggregated runs' )
        plt.figure(); plt.plot(f,s2020_result_db['m'],label='2020');plt.xscale('log');plt.title('dB / m slope for SLR \n South' +', ' + STANDARD + ', aggregated runs' )
        fig2,ax2 = plt.subplots(1,1,figsize=(10,8))        
        ax2, sl_nom ,rl_db_s = \
            SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                p_ax = ax2,
                p_label = '2019',
                p_f_values      = f,
                p_m_values      = s2019_result_db['m'],
                p_track_dist_m  = track_dist,
                p_color         = 'black',
                p_linestyle     ='--')
        ax2, sl_nom ,rl_db_s = \
            SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
                p_ax = ax2,
                p_label = '2020',
                p_f_values      = f,
                p_m_values      = s2020_result_db['m'],
                p_track_dist_m  = track_dist,
                p_color         = 'red',
                p_linestyle     = '-')
        fig2.suptitle(r'$\varepsilon$, assuming CPA reference and '+STANDARD+' standard\n South hydrophone'); plt.xscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Estimated error, $dB ref 1{\mu}Pa^2 / Hz @ 1m$')
        ax2.legend()    
    
    
if COMPARE_P_AND_M:
    
    r2019_unmasked   = SLR_with_transforms.load_concat_arrays('concatenated_data_2019.pkl')
    r2020_unmasked   = SLR_with_transforms.load_concat_arrays('concatenated_data_2020.pkl')
    
    fdir    = _dirs.DIR_RESULT_SLR

    
    for STD in _vars.STANDARDS:
        fname_std = fdir + STD +'_'

        
        r2019   = SLR_with_transforms.mask_data(r2019_unmasked,STANDARD)
        r2020   = SLR_with_transforms.mask_data(r2020_unmasked,STANDARD)
        f       = r2019['Frequency']  
        
        #
        #
        # The y-SLR data
        
        ny2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sy2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        ny2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sy2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        #
        #
        # The x-SLR data
        
        nx2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['North'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sx2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['South'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        nx2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['North'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sx2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['South'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
    
        # South, y axis
        fname = fname_std + 'south_y'
        fig, axs = plt.subplots(2,figsize=(8,6)) # axs[0] is top one
        fig.suptitle('P-value and slope for SLR(y)\n' + STD + ', ' + 'South hydrophone')
        axs[0].plot(f,sy2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,sy2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,sy2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,sy2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value')
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m')
        fig.supxlabel('Frequency (Hz)')
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)
    
        # South, x axis
        fname = fname_std + 'south_x'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        fig.suptitle('P-value and slope for SLR(x)\n' + STD + ', ' + 'South hydrophone')
        axs[0].plot(f,sx2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,sx2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,sx2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,sx2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value')
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m')
        fig.supxlabel('Frequency (Hz)')
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)
    
        # North, y axis
        fname = fname_std + 'north_y'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        fig.suptitle('P-value and slope for SLR(y)\n' + STD + ', ' + 'North hydrophone')
        axs[0].plot(f,ny2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,ny2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,ny2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,ny2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value')
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m')
        fig.supxlabel('Frequency (Hz)')
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)

        # North, x axis
        fname = fname_std + 'north_x'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        fig.suptitle('P-value and slope for SLR(x)\n' + STD + ', ' + 'North hydrophone')
        axs[0].plot(f,nx2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,nx2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,nx2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,nx2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value')
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m')
        fig.supxlabel('Frequency (Hz)')
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)
        

if COMPARE_P_AND_M_SUPPRESS_ID_INFO:
    
    r2019_unmasked   = SLR_with_transforms.load_concat_arrays('concatenated_data_2019.pkl')
    r2020_unmasked   = SLR_with_transforms.load_concat_arrays('concatenated_data_2020.pkl')
    
    fdir    = _dirs.DIR_RESULT_SLR

    
    for STD in _vars.STANDARDS:
        fname_std = fdir + STD +'_'

        
        r2019   = SLR_with_transforms.mask_data(r2019_unmasked,STANDARD)
        r2020   = SLR_with_transforms.mask_data(r2020_unmasked,STANDARD)
        f       = r2019['Frequency']  
        
        #
        #
        # The y-SLR data
        
        ny2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sy2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        ny2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sy2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        #
        #
        # The x-SLR data
        
        nx2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['North'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sx2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2019['X'],
            p_y             = r2019['Y'],
            p_theta         = np.zeros_like(r2019['X']), #not used placeholder
            p_gram          = r2019['South'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        nx2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['North'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
        
        sx2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = r2020['X'],
            p_y             = r2020['Y'],
            p_theta         = np.zeros_like(r2020['X']), #not used placeholder
            p_gram          = r2020['South'],
            p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )
    
        # South, y axis
        fname = fname_std + 'south_y_presn'
        fig, axs = plt.subplots(2,figsize=(8,6)) # axs[0] is top one
        # fig.suptitle('P-value and slope for SLR(y)\n' + STD + ', ' + 'South hydrophone')
        axs[0].plot(f,sy2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,sy2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,sy2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,sy2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value',fontsize=_thesis.FONTSIZE)
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m',fontsize=_thesis.FONTSIZE)
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.FONTSIZE)
        # axs[0].legend(loc='upper left')
        # axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)
    
        # South, x axis
        fname = fname_std + 'south_x_presn'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        # fig.suptitle('P-value and slope for SLR(x)\n' + STD + ', ' + 'South hydrophone')
        axs[0].plot(f,sx2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,sx2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,sx2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,sx2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value',fontsize=_thesis.FONTSIZE)
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m',fontsize=_thesis.FONTSIZE)
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.FONTSIZE)
        # axs[0].legend(loc='upper left')
        # axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)
    
        # North, y axis
        fname = fname_std + 'north_y_presn'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        # fig.suptitle('P-value and slope for SLR(y)\n' + STD + ', ' + 'North hydrophone')
        axs[0].plot(f,ny2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,ny2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,ny2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,ny2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value',fontsize=_thesis.FONTSIZE)
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m',fontsize=_thesis.FONTSIZE)
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.FONTSIZE)
        # axs[0].legend(loc='upper left')
        # axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)

        # North, x axis
        fname = fname_std + 'north_x_presn'
        fig, axs = plt.subplots(2,figsize=(10,8)) # axs[0] is top one
        # fig.suptitle('P-value and slope for SLR(x)\n' + STD + ', ' + 'North hydrophone')
        axs[0].plot(f,nx2019_result_db['p'],label='2019 p-value');
        axs[0].plot(f,nx2020_result_db['p'],label='2020 p-value');
        axs[0].set_ylim(0,0.2)
        axs[0].set_xscale('log');
        axs[1].plot(f,nx2019_result_db['m'],label='2019 slope');
        axs[1].plot(f,nx2020_result_db['m'],label='2020 slope');
        axs[1].set_xscale('log');
        axs[0].set_ylabel('p-value',fontsize=_thesis.FONTSIZE)
        axs[0].axhline(0.05,linewidth=1,color='r') # zero mean line
        axs[1].set_ylabel('Slope, dB / m',fontsize=_thesis.FONTSIZE)
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.FONTSIZE)
        # axs[0].legend(loc='upper left')
        # axs[1].legend(loc='upper left')
        plt.savefig(fname+'.pdf', dpi=300)
        plt.savefig(fname+'.png', dpi=300)


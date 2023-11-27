# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:21:12 2023

@author: Jasper

Make figures for the nonstationary paper

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# PYDAL
import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal.utils

# UWAENVTOOLS
import UWAEnvTools.bathymetry as bathymetry
import UWAEnvTools.locations as locations
from UWAEnvTools.environment import Approximations

paper_figures_directory = r'C:\Users\Jasper\Desktop\papers\MASC_evidence_of_nonstationary_signals_in_shallow_range\figs\\'

SHIP_TRACKS         = False
SPECTRAL_SERIES     = False
RESULTS_Y_TO_Y      = True
if RESULTS_Y_TO_Y : #PICK ONE
    STANDARD        = 'ISO'
    # STANDARD        = 'STANAG'
FREQ_LIST           = [158,55]    


# FIGURE ONE : BATHYMETRY AND SHIP TRACKS:
if SHIP_TRACKS:
    # the raw information / locations needed:
    location = 'Patricia Bay'
    track_rotation_rads = _vars.TRACK_ROTATION_DEG * np.pi / 180
    the_location = locations.Location(location)
    MTH = _vars.MTH
    
    p_dir_spec,run_list = pydal.utils.get_fully_qual_spec_path()
    p_head          = 'X'
    p_speed         = 'X'
    run_list        = pydal.utils.get_run_selection(
        run_list,
        p_type      ='DR',
        p_mth       = 'X',
        p_machine   = 'X',
        p_speed     = p_speed,
        p_head      = p_head,
        p_beam      = 'B')   
    
    # Get the correct bound limits in x and y. This is from Location() so it can go first.
    approx = Approximations()
    cpa_latlon = (the_location.LAT,the_location.LON)
    lat_extent_tuple = the_location.LAT_RANGE_CORRIDOR_TUPLE
    lon_extent_tuple = the_location.LON_RANGE_CORRIDOR_TUPLE
    x_east = approx.latlon_to_xy(cpa_latlon, (cpa_latlon[0],lon_extent_tuple[1])) # most positive ==> east
    x_west = approx.latlon_to_xy(cpa_latlon, (cpa_latlon[0],lon_extent_tuple[0])) #most negative ==> west
    y_north = approx.latlon_to_xy(cpa_latlon, (lat_extent_tuple[1],cpa_latlon[1])) # most positive ==> north
    y_south = approx.latlon_to_xy(cpa_latlon, (lat_extent_tuple[1],cpa_latlon[1])) # most negative  ==> south
    x_lim = x_east[0]
    y_lim = y_north[1]
    
    
    bathy = bathymetry.Bathymetry_CHS_2()
    bathy.read_bathy(the_location.fname_bathy)
    bathy.get_2d_bathymetry_trimmed( #og variable values
                                  p_location_as_object = the_location,
                                  p_num_points_lon = 200,
                                  p_num_points_lat = 200,
                                  # p_lat_delta = 0.00095,
                                  # p_lon_delta = 0.0015,
                                  p_depth_offset = 0)
    # Select from trimming operation above, not the entire bay
    lat = np.array(bathy.lats_selection)
    lon = np.array(bathy.lons_selection)
    z = np.array(bathy.z_selection)
    x,y = bathy.convert_latlon_to_xy_m (
        the_location,
        lat,
        lon
        )
    xr,yr= pydal.utils.rotate(x, y, track_rotation_rads)
    # interpolate the z function over min to max x and y values:
    xlim = 105
    ylim = 105
    xnstep = xlim*2
    ynstep = ylim*2
    x_basis = np.linspace(-1*xlim,xlim,xnstep)
    y_basis = np.linspace(-1*ylim,ylim,ynstep)
    x_target,y_target = np.meshgrid(x_basis,y_basis)
    
    source_points = ( xr , yr )
    xi = ( x_target , y_target )
    z_interp = -1 * interpolate.griddata(
        source_points,
        z,
        xi
        ).T 
    # NOTE THE TRANSPOSE HERE! This puts the X,Y system identical to range.
    # Nominal hydrophone locations are then +/-100,0.
    
    fig,ax  = plt.subplots(1,1,figsize = (10,8));
    extent  = (-1*xlim,xlim,-1*ylim,ylim)
    depths  = ax.imshow(z_interp,extent = extent, origin='lower')
    cbar    = plt.colorbar(depths)
    cbar.set_label('Depth (m)', rotation=270)
    for r in run_list:
        spec_dict,N   = \
            pydal.utils.get_spectrogram_file_as_dict(
                r, p_dir_spec)
        xx          = spec_dict [ 'X' ] 
        yy          = spec_dict [ 'Y' ] 
        if r[:3] == 'DRJ':    
            ax.plot(xx,yy,linestyle='--',color='red')
        if r[:3] == 'DRF':    
            xx,yy= pydal.utils.rotate(xx, yy, track_rotation_rads)
            ax.plot(xx,yy,linestyle='--',color='orange')
    ax.scatter(100,0,marker='X',color='r')
    ax.scatter(-100,0,marker='X',color='r')
    ax.set_xlabel('Range X-coordinate (m)')
    ax.set_ylabel('Range Y-coordinate (m)')
    fig.suptitle('Bathymetry and ship track through Patricia Bay range in local cartesian system')
    
    plt.savefig(fname = paper_figures_directory \
                + r'ship_tracks.eps',
                format='eps',
                dpi = 1200)    
    plt.savefig(fname = paper_figures_directory \
                + r'ship_tracks.pdf',
                format='pdf',
                dpi = 600)
    plt.savefig(fname = paper_figures_directory \
                + r'ship_tracks.png',
                format='png',
                dpi = 1200)
    

if SPECTRAL_SERIES:

    N_AVE           = 11 # for a smoothing window, 9 or 11 should be comparable to 0% overlap timeseries (90% is used still)
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
        p_mth='J',
        p_machine = 'X',
        p_speed = p_speed,
        p_head = p_head)
    
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(run_list[0], dir_spec)
    f_basis           = gram_dict['Frequency']


    for p_freq in FREQ_LIST:
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
        
        delta = [] # in the same order as runs list above, has delta RL (south - north ) for the whole gram
        for runID in run_list:
            gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(runID, dir_spec)
            t.append(gram_dict[ south + '_Spectrogram_Time'])
            xs.append(gram_dict['X'])
            ys.append(gram_dict['Y'])
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
        
        for run,times,x,y,rl_s,rl_n in zip(runs,t,xs,ys,RL_s,RL_n):
            # time basis for sanity checking of west v east runs
            t_loc = np.arange(len(rl_s))
            t_loc = t_loc - np.mean(t_loc) # zero centered, equal on both sides
            t_loc = t_loc / np.max(t_loc) # now scaled to -1 to 1 
            t_loc = t_loc * 100
            # real stuff, sorts by y position only (heading agnostic)
            s = rl_s - np.mean(rl_s)
            s = np.convolve( s , ave_kernel , mode='same')
            n = rl_n - np.mean(rl_n)
            n = np.convolve( n , ave_kernel , mode='same')
            # Y = times - np.mean(times)
            # Y = Y / np.max(Y) 
            # Y = Y * 100
            # NORTH
            # axs[0].plot( t_loc , n, label='North' )#, label = run )
            axs[0].plot( y , n, label='North' )#, label = run )
            # SOUTH
            # axs[1].plot( t_loc , s, label='South' )#, label = run )
            axs[1].plot( y , s, label='South' )#, label = run )
            # DELTA
            # axs[1].plot( y , s - n )#, label = run )
                
        axs[0].axhline(0,linewidth=2) # zero mean line
        axs[1].axhline(0,linewidth=2) # zero mean line
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
        plt.savefig(fname = paper_figures_directory \
                    + r'fig_2_spectral_series'+str(p_freq).zfill(3)+'.eps',
                    format='eps',
                    dpi = 1200)    
        plt.savefig(fname = paper_figures_directory \
                    + r'fig_2_spectral_series'+str(p_freq).zfill(3)+'.pdf',
                    format='pdf',
                    dpi = 600)
        plt.savefig(fname = paper_figures_directory \
                    + r'fig_2_spectral_series'+str(p_freq).zfill(3)+'.png',
                    format='png',
                    dpi = 1200)

if RESULTS_Y_TO_Y:
    
    import SLR_with_transforms
    
    if STANDARD == 'STANAG':
        track_dist = 200;
    if STANDARD == 'ISO':
        track_dist = 114;
    if STANDARD == '50m':
        track_dist = 50;
    
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
    plt.figure(); plt.plot(f,n2020_result_db['m'],label='2019');plt.xscale('log');plt.title('dB / m slope for SLR \n North' +', ' + STANDARD + ', aggregated runs' )
    fig1,ax1 = plt.subplots(1,1,figsize=(10,8))    
    ax1, sl_nom ,rl_db_n2019 = \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = '2019',
            p_f_values      = f,
            p_m_values      = n2019_result_db['m'],
            p_track_dist_m  = track_dist)
    ax1, sl_nom ,rl_db_n2020 = \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = '2020',
            p_f_values      = f,
            p_m_values      = n2020_result_db['m'],
            p_track_dist_m  = track_dist)
    fig1.suptitle('SL_calculated - SL_true, if SL is correct at CPA \n North hydrophone'); plt.xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Estimated error, $dB ref 1{\mu}Pa^2 / Hz @ 1m$')
    ax1.legend()

    plt.figure(); plt.plot(f,s2019_result_db['m'],label='2019');plt.xscale('log');plt.title('dB / m slope for SLR \n South' +', ' + STANDARD + ', aggregated runs' )
    plt.figure(); plt.plot(f,s2020_result_db['m'],label='2019');plt.xscale('log');plt.title('dB / m slope for SLR \n South' +', ' + STANDARD + ', aggregated runs' )
    fig2,ax2 = plt.subplots(1,1,figsize=(10,8))        
    ax2, sl_nom ,rl_db_s = \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax2,
            p_label = '2019',
            p_f_values      = f,
            p_m_values      = s2019_result_db['m'],
            p_track_dist_m  = track_dist)
    ax2, sl_nom ,rl_db_s = \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax2,
            p_label = '2020',
            p_f_values      = f,
            p_m_values      = s2020_result_db['m'],
            p_track_dist_m  = track_dist)
    fig2.suptitle('SL_calculated - SL_true, if SL is correct at CPA \n South hydrophone'); plt.xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Estimated error, $dB ref 1{\mu}Pa^2 / Hz @ 1m$')
    ax2.legend()

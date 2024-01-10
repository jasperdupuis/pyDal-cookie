# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:21:12 2023

@author: Jasper

Make figures for the nonstationary paper,
save as PNG, PDF, EPS Formats.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# PYDAL
import pydal.models.SLR_with_transforms as SLR_with_transforms
import pydal.utils
# PYDAL CONSTANTS
import pydal._variables as _vars
import pydal._directories_and_files as _dirs

# UWAENVTOOLS
import UWAEnvTools.bathymetry as bathymetry
import UWAEnvTools.locations as locations
from UWAEnvTools.environment import Approximations

paper_figures_directory = r'C:\Users\Jasper\Desktop\papers\MASC_evidence_of_nonstationary_signals_in_shallow_range\figs\\'

SHIP_TRACKS                 = False
SPECTRA_NONZERO             = False
ZEROMEAN_SPECTRAL_SERIES    = False
RESULTS_Y_TO_Y              = True

FREQ_LIST           = [155]    

# Tracking FIGNUMBER
if SHIP_TRACKS:                                     FIGNUMBER = 1
if not (SHIP_TRACKS):                               FIGNUMBER = 2
if not (SHIP_TRACKS and SPECTRA_NONZERO):           FIGNUMBER = 3
if not (SHIP_TRACKS and SPECTRA_NONZERO and ZEROMEAN_SPECTRAL_SERIES): FIGNUMBER = 3 + len(FREQ_LIST)

# Plotting variables
# DPI                 = 900
DPI                 = 900
FIGSCALE            = 1
FIGSIZE             = (FIGSCALE * 3.35,FIGSCALE * 3.35)
FIGSIZE_SPEC_TSERIES= (FIGSCALE * 3.35,FIGSCALE * 5)
FONT_FAM            ='serif' #or non-serif, check journal (JASA)
SIZE_XTICK          ='x-small'
SIZE_YTICK          ='x-small'
SIZE_TITLE          = 8
SIZE_AX_LABELS      = 8
SIZE_SCATTER_DOT    = 0.5

plt.rc('font',family=FONT_FAM)
plt.rc('xtick', labelsize=SIZE_XTICK)
plt.rc('ytick', labelsize=SIZE_YTICK)

COLOR_DICTIONARY = { '03' : 'black',
                    '05' : 'red',
                    '07' : 'blue',
                    '08' : 'cyan',
                    '09' : 'yellow',
                    '11' : 'green',
                    '13' : 'cyan',
                    '15' : 'black',
                    '17' : 'red',
                    '19' : 'blue',
                    '20' : 'blue'}


def plot_tracks_bathy(p_fignumber):
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
    
    fig,ax  = plt.subplots(1,1,figsize = FIGSIZE);
    extent  = (-1*xlim,xlim,-1*ylim,ylim)
    depths  = ax.imshow(z_interp,extent = extent, origin='lower')
    cbar    = plt.colorbar(depths)
    for r in run_list:
        spec_dict,N   = \
            pydal.utils.get_spectrogram_file_as_dict(
                r, p_dir_spec)
        xx          = spec_dict [ 'X' ] 
        yy          = spec_dict [ 'Y' ] 
        if r[:3] == 'DRJ':    
            ax.plot(xx,yy,linestyle='--',color='black')
        if r[:3] == 'DRF':    
            xx,yy= pydal.utils.rotate(xx, yy, track_rotation_rads)
            ax.plot(xx,yy,linestyle='-.',color='black')
    ax.scatter(100,0,marker='X',color='r')
    ax.scatter(-100,0,marker='X',color='r')
    ax.set_xlabel('Range X-coordinate (m)')
    ax.set_ylabel('Range Y-coordinate (m)')
    cbar.set_label('Depth (m)', rotation=270, labelpad = 15)
    # fig.suptitle('Bathymetry and ship track through Patricia Bay range in local cartesian system')
    figname = r'fig_' + str(p_fignumber).zfill(2) + r'_ship_tracks'
    p_fignumber = p_fignumber + 1
    plt.savefig(fname = paper_figures_directory \
                + figname +'.eps',
                bbox_inches='tight',
                format='eps',
                dpi = DPI)    
    plt.savefig(fname = paper_figures_directory \
                + figname +'.pdf',
                bbox_inches='tight',
                format='pdf',
                dpi = DPI)
    plt.savefig(fname = paper_figures_directory \
                + figname + '.png',
                bbox_inches='tight',
                format='png',
                dpi = DPI)
    return p_fignumber


def plot_spectral_series(p_fignumber,p_freq,p_N_ave,ylims=[40,100]):
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
    
    ave_kernel = np.ones( p_N_ave ) / p_N_ave
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=FIGSIZE_SPEC_TSERIES)
    
    for run,times,x,y,rl_s,rl_n in zip(runs,t,xs,ys,RL_s,RL_n):
        if not( len(y) == len(rl_s) ): 
            continue
        # time basis for sanity checking of west v east runs
        t_loc = np.arange(len(rl_s))
        t_loc = t_loc - np.mean(t_loc) # zero centered, equal on both sides
        t_loc = t_loc / np.max(t_loc) # now scaled to -1 to 1 
        t_loc = t_loc * 100
        # real stuff, sorts by y position only (heading agnostic)
        rl_s = np.convolve( rl_s , ave_kernel , mode='same')
        rl_n = np.convolve( rl_n , ave_kernel , mode='same')
        # NORTH
        # axs[0].plot( t_loc , n, label='North' )#, label = run )
        ax.plot( rl_n[6:-6] , y[6:-6],  
                       color=COLOR_DICTIONARY[run[6:8]],
                       label='North' )#, label = run )
        # SOUTH
        # axs[1].plot( t_loc , s, label='South' )#, label = run )
        # axs[1].scatter( s , y, 
        #                marker = '.', 
        #                s = SIZE_SCATTER_DOT, 
        #                color='black',
        #                label='South' )#, label = run )
        # DELTA
        # axs[1].plot( y , s - n )#, label = run )
        
    ax.xaxis.set_label_position('top') 
    fig.supxlabel('Spectral time series, dB ref 1 ${\mu}Pa^2 / Hz$', fontsize=SIZE_AX_LABELS)
    fig.supylabel('Y-position in range X-Y system (m)', fontsize=SIZE_AX_LABELS)
    plt.tight_layout()
    figname = r'fig_' + str(p_fignumber).zfill(2) + r'_spectral_series'+str(p_freq).zfill(3)
    p_fignumber = p_fignumber + 1
    plt.savefig(fname = paper_figures_directory \
                + figname +'.eps',
                bbox_inches='tight',
                format='eps',
                dpi = DPI)    
    plt.savefig(fname = paper_figures_directory \
                + figname +'.pdf',
                bbox_inches='tight',
                format='pdf',
                dpi = DPI)
    plt.savefig(fname = paper_figures_directory \
                + figname + '.png',
                bbox_inches='tight',
                format='png',
                dpi = DPI)
    return p_fignumber
 
   
def plot_zeromean_spectral_series(p_fignumber,p_freq,p_N_ave):
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
        p_mth='X',
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
    
    ave_kernel = np.ones( p_N_ave ) / p_N_ave
    fig, axs = plt.subplots(nrows = 1, ncols=2, figsize=FIGSIZE_SPEC_TSERIES)
    
    for run,times,x,y,rl_s,rl_n in zip(runs,t,xs,ys,RL_s,RL_n):
        # time basis for sanity checking of west v east runs
        if not ( len(y) == len(rl_s)) : 
            continue # Fucked up run.
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
        axs[0].scatter( n , y, 
                       marker = '.', 
                       s = SIZE_SCATTER_DOT, 
                       color='black',
                       label='North' )#, label = run )
        # SOUTH
        # axs[1].plot( t_loc , s, label='South' )#, label = run )
        axs[1].scatter( s , y, 
                       marker = '.', 
                       s = SIZE_SCATTER_DOT, 
                       color='black',
                       label='South' )#, label = run )
            
    axs[0].axvline(0,color='white',linestyle='--',linewidth=1) # zero mean line
    axs[1].axvline(0,color='white',linestyle='--',linewidth=1) # zero mean line
    axs[0].set_xlabel(r'North', fontsize=SIZE_AX_LABELS)
    axs[0].xaxis.set_label_position('top') 
    axs[1].set_xlabel(r'South', fontsize=SIZE_AX_LABELS)
    axs[1].xaxis.set_label_position('top') 
    fig.supxlabel('Zero-mean spectral time series, dB ref 1 ${\mu}Pa^2 / Hz$', fontsize=SIZE_AX_LABELS)
    fig.supylabel('Y-position in range X-Y system (m)', fontsize=SIZE_AX_LABELS)
    plt.tight_layout()
    figname = r'fig_' + str(p_fignumber).zfill(2) + r'_zeromean_spectral_series'+str(p_freq).zfill(3)
    p_fignumber = p_fignumber + 1
    plt.savefig(fname = paper_figures_directory \
                + figname +'.eps',
                bbox_inches='tight',
                format='eps',
                dpi = DPI)    
    plt.savefig(fname = paper_figures_directory \
                + figname +'.pdf',
                bbox_inches='tight',
                format='pdf',
                dpi = DPI)
    plt.savefig(fname = paper_figures_directory \
                + figname + '.png',
                bbox_inches='tight',
                format='png',
                dpi = DPI)
    return p_fignumber


def plot_fig_result_Y_to_Y(p_fignumber,p_hydro = 'North',p_f_lims=(30,300)):
    
    track_dist_stanag   = 200;
    track_dist_iso      = 114;

    r2019   = SLR_with_transforms.load_concat_arrays('concatenated_data_2019.pkl')
    r2020   = SLR_with_transforms.load_concat_arrays('concatenated_data_2020.pkl')

    r2019_iso   = SLR_with_transforms.mask_data(r2019,'ISO')
    r2020_iso   = SLR_with_transforms.mask_data(r2020,'ISO')
    r2019_sta   = SLR_with_transforms.mask_data(r2019,'STANAG')
    r2020_sta   = SLR_with_transforms.mask_data(r2020,'STANAG' )


    f           = r2019['Frequency']  
    f_ind_low   = pydal.utils.find_target_freq_index(p_f_lims[0],f)
    f_ind_high  = pydal.utils.find_target_freq_index(p_f_lims[1],f)

    iso_2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
        p_x             = r2019_iso['X'],
        p_y             = r2019_iso['Y'],
        p_theta         = np.zeros_like(r2019_iso['X']), #not used placeholder
        p_gram          = r2019_iso[p_hydro],
        p_x_transform   = pydal.data_transforms.x_transform_y_only,
        # p_x_transform   = pydal.data_transforms.x_transform_x_only,
        p_y_transform   = pydal.data_transforms.no_2d_transform
        )

    sta_2019_result_db     = SLR_with_transforms.SLR_with_y_transform(
        p_x             = r2019_sta['X'],
        p_y             = r2019_sta['Y'],
        p_theta         = np.zeros_like(r2019_sta['X']), #not used placeholder
        p_gram          = r2019_sta[p_hydro],
        p_x_transform   = pydal.data_transforms.x_transform_y_only,
        # p_x_transform   = pydal.data_transforms.x_transform_x_only,
        p_y_transform   = pydal.data_transforms.no_2d_transform
        )

    iso_2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
        p_x             = r2020_iso['X'],
        p_y             = r2020_iso['Y'],
        p_theta         = np.zeros_like(r2020_iso['X']), #not used placeholder
        p_gram          = r2020_iso[p_hydro],
        p_x_transform   = pydal.data_transforms.x_transform_y_only,
        # p_x_transform   = pydal.data_transforms.x_transform_x_only,
        p_y_transform   = pydal.data_transforms.no_2d_transform
        )

    sta_2020_result_db     = SLR_with_transforms.SLR_with_y_transform(
        p_x             = r2020_sta['X'],
        p_y             = r2020_sta['Y'],
        p_theta         = np.zeros_like(r2020_sta['X']), #not used placeholder
        p_gram          = r2020_sta[p_hydro],
        p_x_transform   = pydal.data_transforms.x_transform_y_only,
        # p_x_transform   = pydal.data_transforms.x_transform_x_only,
        p_y_transform   = pydal.data_transforms.no_2d_transform
        )

    fig1,ax1 = plt.subplots(1,1,figsize=FIGSIZE)    
    # ISO
    ax1, _ ,_ = \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = 'ISO-19',
            p_f_values      = f[f_ind_low:f_ind_high],
            p_m_values      = iso_2019_result_db['m'][f_ind_low:f_ind_high],
            p_track_dist_m  = track_dist_iso,
            p_color         = 'black',
            p_linestyle     = '-')
    ax1, _, _= \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = 'ISO-20',
            p_f_values      = f[f_ind_low:f_ind_high],
            p_m_values      = iso_2020_result_db['m'][f_ind_low:f_ind_high],
            p_track_dist_m  = track_dist_iso,
            p_color         = 'red',
            p_linestyle     = '-')
    # STANAG
    ax1, _, _= \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = 'NATO-19',
            p_f_values      = f[f_ind_low:f_ind_high],
            p_m_values      = sta_2019_result_db['m'][f_ind_low:f_ind_high],
            p_track_dist_m  = track_dist_stanag,
            p_color         = 'cyan',
            p_linestyle     = '-')
    ax1, _, _= \
        SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
            p_ax = ax1,
            p_label = 'NATO-20',
            p_f_values      = f[f_ind_low:f_ind_high],
            p_m_values      = sta_2020_result_db['m'][f_ind_low:f_ind_high],
            p_track_dist_m  = track_dist_stanag,
            p_color         = 'green',
            p_linestyle     = '-')

    #Epsilon is how much higher SPL is than a constant k value after averaging using the slope found from SLR.
    # fig1.suptitle(r'$\varepsilon$'+', North hydrophone\nCPA reference and '+STANDARD+' standard',fontsize=SIZE_AX_LABELS); 

    plt.xscale('log')
    ax1.set_ylim((0,2.5))
    ax1.set_xlabel('Frequency (Hz)',fontsize=SIZE_AX_LABELS)
    ax1.set_ylabel(r'$\varepsilon$'+', dB ref $1{\mu}Pa^2m^2 / Hz$',fontsize=SIZE_AX_LABELS)
    ax1.legend(loc='upper right')
    # ax1.legend(loc='upper center', mode = 'expand', bbox_to_anchor=(0,1.14, 1,0.1),
    #        fancybox=True, shadow=True, ncol=2)
    figname = r'fig_' + str(p_fignumber ).zfill(2) + r'_'+p_hydro.lower()+'_error_linear_model'
    p_fignumber  = p_fignumber  + 1
    plt.savefig(fname = paper_figures_directory \
                + figname +'.eps',
                bbox_inches='tight',
                format='eps',
                dpi = DPI)    
    plt.savefig(fname = paper_figures_directory \
                + figname +'.pdf',
                bbox_inches='tight',
                format='pdf',
                dpi = DPI)
    plt.savefig(fname = paper_figures_directory \
                + figname +'.png',
                bbox_inches='tight',
                format='png',
                dpi = DPI)
    plt.close('all')

    return p_fignumber

# FIGURE ONE : BATHYMETRY AND SHIP TRACKS:
if SHIP_TRACKS:
    FIGNUMBER = plot_tracks_bathy(FIGNUMBER)
    plt.close('all')


if SPECTRA_NONZERO:
    freq = 55
    FIGNUMBER = plot_spectral_series(FIGNUMBER,freq,p_N_ave = 11)
    plt.close('all')


if ZEROMEAN_SPECTRAL_SERIES:

    # for 90% overlap:
    N_AVE_ZMEAN = 5
    for freq in FREQ_LIST:
        FIGNUMBER = plot_zeromean_spectral_series(FIGNUMBER,freq,N_AVE_ZMEAN)
        plt.close('all')


if RESULTS_Y_TO_Y:
    #total freq basis:
    # Don't care to show STANAG over the larger frequency range.
    # FIGNUMBER = plot_fig_result_Y_to_Y(FIGNUMBER,'ISO',p_f_lims=(10,10000))
    # plt.close('all')
    # FIGNUMBER = plot_fig_result_Y_to_Y(FIGNUMBER,'STANAG',p_f_lims=(10,10000))
    # plt.close('all')
    
    #30 to 300 only
    
    # FIGNUMBER, n_2019iso,s_2019iso,n_2020iso,s_2020iso= \
    #     plot_fig_result_Y_to_Y(FIGNUMBER,'ISO',p_f_lims=(30,300))
    # plt.close('all')
    # FIGNUMBER, n_2019sta,s_2019sta,n_2020sta,s_2020sta= \
    #     plot_fig_result_Y_to_Y(FIGNUMBER,'STANAG',p_f_lims=(30,300))
    # plt.close('all')

    FIGNUMBER = plot_fig_result_Y_to_Y(FIGNUMBER,'North',p_f_lims=(20,1000))
    plt.close('all')
    FIGNUMBER = plot_fig_result_Y_to_Y(FIGNUMBER,'South',p_f_lims=(20,1000))
    plt.close('all')


    
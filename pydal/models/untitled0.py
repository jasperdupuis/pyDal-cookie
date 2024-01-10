# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:29:55 2023

@author: Jasper
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


p_f_lims = (30,300)
p_hydro='North'
p_fignumber = 1

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
        p_linestyle     = '--')
ax1, _, _= \
    SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
        p_ax = ax1,
        p_label = 'ISO-20',
        p_f_values      = f[f_ind_low:f_ind_high],
        p_m_values      = iso_2020_result_db['m'][f_ind_low:f_ind_high],
        p_track_dist_m  = track_dist_iso,
        p_color         = 'green',
        p_linestyle     = '-')
# STANAG
ax1, _, _= \
    SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
        p_ax = ax1,
        p_label = 'NATO-19',
        p_f_values      = f[f_ind_low:f_ind_high],
        p_m_values      = sta_2019_result_db['m'][f_ind_low:f_ind_high],
        p_track_dist_m  = track_dist_stanag,
        p_color         = 'black',
        p_linestyle     = 'dotted')
ax1, _, _= \
    SLR_with_transforms.compare_SL_nominal_vs_RL_slope_implied(
        p_ax = ax1,
        p_label = 'NATO-20',
        p_f_values      = f[f_ind_low:f_ind_high],
        p_m_values      = sta_2020_result_db['m'][f_ind_low:f_ind_high],
        p_track_dist_m  = track_dist_stanag,
        p_color         = 'green',
        p_linestyle     = '-.')

#Epsilon is how much higher SPL is than a constant k value after averaging using the slope found from SLR.
# fig1.suptitle(r'$\varepsilon$'+', North hydrophone\nCPA reference and '+STANDARD+' standard',fontsize=SIZE_AX_LABELS); 

plt.xscale('log')
ax1.set_xlabel('Frequency (Hz)',fontsize=SIZE_AX_LABELS)
ax1.set_ylabel(r'$\varepsilon$'+', dB ref $1{\mu}Pa^2m^2 / Hz$',fontsize=SIZE_AX_LABELS)
ax1.legend(loc='upper right')
figname = r'fig_' + str(p_fignumber ).zfill(2) + r'_'+p_hydro.lower()+'_error_linear_model'
p_fignumber  = p_fignumber  + 1
# plt.savefig(fname = paper_figures_directory \
#             + figname +'.eps',
#             bbox_inches='tight',
#             format='eps',
#             dpi = DPI)    
# plt.savefig(fname = paper_figures_directory \
#             + figname +'.pdf',
#             bbox_inches='tight',
#             format='pdf',
#             dpi = DPI)
# plt.savefig(fname = paper_figures_directory \
#             + figname +'.png',
#             bbox_inches='tight',
#             format='png',
#             dpi = DPI)
# plt.close('all')

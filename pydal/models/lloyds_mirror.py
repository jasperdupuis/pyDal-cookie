# -*- coding: utf-8 -*-

"""
Created on Fri Oct 13 16:04:26 2023

@author: Jasper

see https://acousticstoday.org/wp-content/uploads/2017/07/Article_2of3_from_ATCODK_5_2.pdf

"""

import numpy as np
import matplotlib.pyplot as plt

import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis

R       = 100
# R       = np.arange(0,1000)
# f       = 70
f       = np.arange(10,1000)
z_s     = 3
z_r_s   = 41
z_r_n   = 25
c       = 1500
lamb    = c/f
k       = 2 * np.pi / lamb
theta_n   = np.arctan(z_r_n / R)
theta_s   = np.arctan(z_r_s / R)

"""

Venditts ten challenges to measure URN paper

Equations 1 and 2.

in the paper, d == z_s (not bottom depth)

"""

# # TL = 20 log ( sin ( kd sin ( theta ) ) ) + 20 log r # Directly from paper
# theta   = np.arctan(z_r / R)
# TL      = 20 * np.log10 ( np.sin( k * z_s * np.sin(theta))) #-20*np.log10(R)
# plt.figure();plt.plot(R,TL);plt.xscale('log');plt.xlabel('Distance (m)');plt.title('Vendittis LME eq1,' + str(f) + ' Hz')
# plt.plot(R,-20*np.log10(R),label='20 Log R')
# plt.legend()

# TL_simple = 20 * np.log10 ( 2 * k * z_s * np.sin(theta)) - 20*np.log10(R)
# plt.figure();plt.plot(R,TL_simple);plt.xscale('log');plt.xlabel('Distance (m)');plt.title('Vendittis LME eq 2,' + str(f) + ' Hz')
# # plt.figure();plt.plot(R,TL_simple);plt.xlabel('Distance (m)');plt.title('Vendittis LME eq 2,' + str(f) + ' Hz')
# plt.plot(R,-20*np.log10(R),label='20 Log R')
# plt.legend()


"""
COA page 15, equation 1.6
Additional expansions in equations 1.7

Explicit formula for p(r,z) using complex exponentials
"""
R1_s      = np.sqrt( R**2 + ( z_r_s - z_s ) **2 )
R2_s      = np.sqrt( R**2 + ( z_r_s + z_s ) **2 )
R1_n      = np.sqrt( R**2 + ( z_r_n - z_s ) **2 )
R2_n      = np.sqrt( R**2 + ( z_r_n + z_s ) **2 )

p_r_z_s   = (np.exp( 1j * k * R1_s ) / R1_s) - (np.exp( 1j * k * R2_s ) / R2_s)
p2_s      = np.abs(p_r_z_s)**2
p2db_s    = 10*np.log10(p2_s)

p_r_z_n   = (np.exp( 1j * k * R1_n ) / R1_n) - (np.exp( 1j * k * R2_n ) / R2_n)
p2_n      = np.abs(p_r_z_n)**2
p2db_n    = 10*np.log10(p2_n)


plt.figure()
plt.plot(f,p2db_s,label='South hydrophone');
plt.plot(f,p2db_n,label='North hydrophone');
plt.xscale('log');
plt.xlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS);
plt.ylabel('LME @ 100 m, dB ref $1{\mu}Pa^2 / Hz$',fontsize=_thesis.SIZE_AX_LABELS)
plt.legend(loc='lower left')

figname =  'lloyds_mirror'
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
# plt.close('all')

"""

Venditts ten challenges to measure URN paper

Good plot script:

"""
"""
theta   = np.arctan(z_s / R)
k       = 2 * np.pi / lamb
TL      = 20 * np.log10 ( np.sin( k * z_s * np.sin(theta))) + 20*np.log10(R)
plt.figure(figsize=_thesis.FIGSIZE);
plt.plot(f,TL);
plt.xscale('log');
plt.xlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS);
plt.ylabel('LME @ 100 m, dB ref $1{\mu}Pa^2 / Hz$',fontsize=_thesis.SIZE_AX_LABELS)

figname =  'lloyds_mirror'
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = _dirs.DIR_RESULT_LME  \
            + figname +'.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
# plt.close('all')
"""

"""
Look at a gram for a run i choose at random from the run_list variable.
"""

# import pydal.utils

# dir_spec ,run_list  = pydal.utils.get_fully_qual_spec_path()
# p_runID             = run_list[35]
# gram_dict,N         = pydal.utils.get_spectrogram_file_as_dict(p_runID, dir_spec)    

# n                   = gram_dict['North_Spectrogram'][:50000,:]
# ndb                 = 10 * np.log10 (n) 
# s                   = gram_dict['South_Spectrogram'][:50000,:]
# sdb                 = 10 * np.log10 (s) 


# plt.figure();plt.imshow(ndb,aspect='auto',origin='lower');plt.title('North Gram');plt.colorbar()
# plt.figure();plt.imshow(sdb,aspect='auto',origin='lower');plt.title('South Gram');plt.colorbar()


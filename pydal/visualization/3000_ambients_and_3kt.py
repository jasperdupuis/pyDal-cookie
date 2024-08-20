# -*- coding: utf-8 -*-
"""

Visualize 2019 and 2020 ambient with 2019 quietest runs from fmin to ~1kHz

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pydal.utils

import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis

FMAX                = 500
AMB_LINESKIP        = 73
DSR_LINESKIP_2019    = 61
DSR_LINESKIP_2020    = 73

DIR_RANGE_DSR_RESULT_2019 = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN'
DIR_RANGE_DSR_RESULT_2020 = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data\ES0453_MOOSE_SPC_DYN'


def plot_ambient_with_bottom_and_top_speeds(p_dir,ax,DSR_LINESKIP):
    list_ambs = os.listdir(p_dir)
    list_ambs = [x for x in list_ambs if 'AM' in x]
    list_ambs = [x for x in list_ambs if 'NB' in x]
    list_ambs_n = [x for x in list_ambs if 'Nhyd' in x]
    list_ambs_s = [x for x in list_ambs if 'Shyd' in x]
    
    #the ambients
    for f in list_ambs_n:
        if '007_000' in f: continue #bad run
        if '006_000' in f: continue #bad run
        fname       = p_dir + r'\\' + f
        df          = pd.read_csv(fname,skiprows=AMB_LINESKIP)
        columns     = df.columns
        freq        = df[columns[0]]
        psd         = df[columns[1]]
        max_index   = pydal.utils.find_target_freq_index(FMAX, freq) 
        ax.plot(freq[:max_index],psd[:max_index]-40,linestyle='--')
    
    list_runs = os.listdir(p_dir)
    list_runs = [x for x in list_runs if 'RUN' in x]
    list_runs = [x for x in list_runs if 'NB' in x]
    list_runs_n = [x for x in list_runs if 'Nhyd' in x]
    list_runs_s = [x for x in list_runs if 'Shyd' in x]
    
    # the dynamic runs
    for f in list_runs_n:
        fname       = p_dir + r'\\' + f
        # check for lowest speeds:
        speed = 10
        with open(fname) as file:
            for line in file:
                if r'Speed Actual' in line:
                    string = line.split(' ')[-1]
                    try:
                        speed = float(string)
                    except:
                        continue #range fuckup
                if r'Corr Hyd' in line:
                    string = line.split(' ')[-1]
                    correction = float(string)
                    break
        if speed < 4.5 or speed > 18:
            if speed == 0.0: continue
            # df          = pd.read_csv(fname,skiprows=DSR_LINESKIP,encoding='UTF-8')
            df          = pd.read_csv(fname,skiprows=DSR_LINESKIP,encoding='latin-1')
            columns     = df.columns
            freq        = df[columns[0]]
            psd         = df[columns[1]]
            max_index   = pydal.utils.find_target_freq_index(FMAX, freq) 
            ax.plot(freq[:max_index],psd[:max_index]-correction,label=str(speed) + ' knots')
    return ax

#
# 2019 first
#
p_dir = DIR_RANGE_DSR_RESULT_2019

fig,ax = plt.subplots(figsize=_thesis.WIDE_FIG_SIZE)
ax = plot_ambient_with_bottom_and_top_speeds(p_dir,ax,DSR_LINESKIP_2019)

fig.supylabel('Hydrophone received PSD, dB ref 1 ${\mu}Pa^2 / Hz$', fontsize=_thesis.SIZE_AX_LABELS)
fig.supxlabel('Frequency, Hz', fontsize=_thesis.SIZE_AX_LABELS)
ax.legend()
plt.tight_layout()

figname = r'ambient_and_run_comparison_2019'
fname = _dirs.DIR_RESULT_AMBIENTS + figname
plt.savefig(fname = fname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = fname + '.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = fname + '.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
plt.close('all')

#
# Now 2020:
#
   
p_dir = DIR_RANGE_DSR_RESULT_2020

fig,ax = plt.subplots(figsize=_thesis.WIDE_FIG_SIZE)
ax = plot_ambient_with_bottom_and_top_speeds(p_dir,ax,DSR_LINESKIP_2020)

fig.supylabel('Hydrophone received PSD, dB ref 1 ${\mu}Pa^2 / Hz$', fontsize=_thesis.SIZE_AX_LABELS)
fig.supxlabel('Frequency, Hz', fontsize=_thesis.SIZE_AX_LABELS)
ax.legend()
plt.tight_layout()

figname = r'ambient_and_run_comparison_2020'
fname = _dirs.DIR_RESULT_AMBIENTS + figname
plt.savefig(fname = fname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = fname + '.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = fname + '.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
plt.close('all')

   
    


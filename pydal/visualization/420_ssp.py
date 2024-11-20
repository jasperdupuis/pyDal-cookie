# -*- coding: utf-8 -*-
"""

Display raw SSP in one image.

"""


import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import pydal._thesis_constants as _thesis
import pydal._directories_and_files as _dirs

SKIPROWS = 28

dir_ssp = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\Pat Bay CTDs\\'

fnames = os.listdir(dir_ssp)

fn_2019 = [ f for f in fnames if '2019' in f]
fn_2020 = [ f for f in fnames if '2020' in f]

fig,ax = plt.subplots(1,figsize = _thesis.FIGSIZE)

# f = fn_2019[1]
for f in fn_2019:
    fname   = dir_ssp + f
    df      = pd.read_csv(fname,skiprows=SKIPROWS)
    d       = df[df.columns[1]]
    ssp     = df[df.columns[-2]]
    ax.scatter(ssp,d,marker='2',color='red',label='July 2019')

for f in fn_2020:
    fname   = dir_ssp + f
    df      = pd.read_csv(fname,skiprows=SKIPROWS)
    d       = df[df.columns[1]]
    ssp     = df[df.columns[-2]]
    ax.scatter(ssp,d,marker='1',color='blue',label='February 2020')

plt.legend(['July 2019','February 2020'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('red')
leg.legend_handles[1].set_color('blue')

ax.invert_yaxis()
fig.supxlabel('Soundspeed, m/s',fontsize=_thesis.SIZE_AX_LABELS)
fig.supylabel('Depth, m',fontsize = _thesis.SIZE_AX_LABELS)
plt.grid(which='both')

figname =  'SSP'
plt.savefig(fname = _dirs.DIR_RESULT_SSP\
            + figname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = _dirs.DIR_RESULT_SSP\
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = _dirs.DIR_RESULT_SSP\
            + figname +'.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
plt.close('all')






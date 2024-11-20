# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis

import sys
sys.path.insert(1, r'C:\Users\Jasper\Documents\Repo\pyDal\UWAEnvTools')
import UWAEnvTools

N_lats = 1000
N_lons = 1000
LOCATION = 'Patricia Bay'
the_location = UWAEnvTools.locations.Location(LOCATION) 


lat_tuple = the_location.LAT_EXTENT_TUPLE
lon_tuple = the_location.LON_EXTENT_TUPLE


bathymetry = UWAEnvTools.bathymetry.Bathymetry_CHS_2()
bathymetry.read_bathy(the_location.fname_bathy)
bathymetry.sub_select_by_latlon(
    p_lat_extent_tuple = lat_tuple,
    p_lon_extent_tuple = lon_tuple) #has default values for NS already

bathymetry.interpolate_bathy()


the_location.LAT_EXTENT_TUPLE = lat_tuple
the_location.LON_EXTENT_TUPLE = lon_tuple
bathymetry.plot_bathy(
    p_location  = the_location, 
    p_N_LAT_PTS = N_lats, 
    p_N_LON_PTS = N_lons);plt.show()

plt.xlabel('Y-coordinate (m)', fontsize = _thesis.FONTSIZE)
plt.xlabel('X-coordinate (m)', fontsize = _thesis.FONTSIZE)


figname =  'pat_bay_bathymetry'
plt.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.eps',
            bbox_inches='tight',
            format='eps',
            dpi = _thesis.DPI)    
plt.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.pdf',
            bbox_inches='tight',
            format='pdf',
            dpi = _thesis.DPI)
plt.savefig(fname = _dirs.DIR_THESIS_IMGS \
            + figname +'.png',
            bbox_inches='tight',
            format='png',
            dpi = _thesis.DPI)
plt.close('all')





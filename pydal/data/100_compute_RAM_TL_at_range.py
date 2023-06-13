# -*- coding: utf-8 -*-
"""

This module relies on the earlier work captured in import UWAEnvTools

"""


import numpy as np
import matplotlib.pyplot as plt
import time

import pydal._variables as _vars
import pydal._directories_and_files as _dirs

#my modules
import UWAEnvTools
from UWAEnvTools.singleTL.RAM import compute_RAM_corridor_to_hyd
from UWAEnvTools.locations import Location


if __name__ == '__main__':
    # freqs = np.arange(50) + 600     # 600 to 649
    # freqs = np.arange(50) + 650   # 650 to 699
    # freqs = np.arange(50) + 700   # 700 to 749
    # freqs = np.arange(50) + 750   # 750 to 799
    # freqs = np.arange(50) + 800   # 800 to 849    
    # freqs = np.arange(50) + 850   # 850 to 899
    
    n_lat_pts = 100
    n_lon_pts = 100
    location = 'Patricia Bay'
    the_location = Location(location) 

    dir_RAM = _dirs.DIR_RAM_DATA
    delta_r_RAM = _vars.RAM_DELTA_R
    line_or_point = _vars.RAM_RESULTS_POINT_OR_LINE
    
    for freq in freqs:
        print ('Starting freq '+str(freq)+ ' at Pat Bay...')
        start = time.time()
        s1 , n1 = compute_RAM_corridor_to_hyd(
            freq,
            n_lat_pts,
            n_lon_pts,
            dir_RAM,
            delta_r_RAM,
            line_or_point)
        end = time.time()
        print('...computation time for freq ' + str(freq)+ ' : ' + str(int(end-start))+ ' seconds')
# -*- coding: utf-8 -*-
"""

Finally!

Merge the two different data products, synthetic and measured data from the 
spectrogram hdf5 files and the RAM TL model CSVs

Accomplish this by inserting the RAM TL data in to the existing
spectrogram hdf5 files that are output from script # 010 using the provided
XY coordinates and the recovered TL(f,X,Y) function


"""

import time
import numpy as np
import h5py as h5

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs

from UWAEnvTools.singleTL import RAM
from UWAEnvTools.singleTL.RAM import \
    interpolate_TL_over_XY_set_multi_f

    
if __name__ == '__main__':
    p_hydro         = 'SOUTH'
    p_dir_RAM       = _dirs.DIR_RAM_DATA
    p_dir_spec      = _dirs.DIR_SPECTROGRAM
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    p_dir_spec      = p_dir_spec + dir_spec_subdir + '\\'
    
    
    freqs_RAM = np.arange(_vars.RAM_F_MIN_AVAIL,_vars.RAM_F_MAX_AVAIL)
    freqs_RAM = [ f for f in freqs_RAM if f not in _vars.RAM_F_FAILS]
    
    run_list = pydal.utils.get_all_runs_in_dir(p_dir_spec)
    
    # runID = run_list[0]
    for runID in run_list:
        spec_dict = \
            pydal.utils.load_target_spectrogram_data(
                runID, p_dir_spec)
        
        start = time.time()
        
        runID_RAM_TLs = \
            RAM.interpolate_TL_over_XY_set_multi_f(
                p_freq_targets  = freqs_RAM,
                p_f_basis       = spec_dict['Frequency'], 
                p_gram_x        = spec_dict['X'], 
                p_gram_y        = spec_dict['Y'],
                p_dir_RAM   = p_dir_RAM,
                p_hydro     = p_hydro)
        
        end = time.time()    
        print(runID + ' computation time: ' + str(end-start))
        
        fname = p_dir_spec + '\\' + runID + r'_data_timeseries.hdf5' 
        
        gname = p_hydro.capitalize() + '_RAM_TL_interpolations'
        h = h5.File(fname,mode='a')
        try:
            group = h.create_group(gname )
        except:
            print (runID + ' already had ' + p_hydro.capitalize() +' RAM TL interpolation done, overwriting it.')
            del h[ gname ]
            group = h.create_group(gname)
        for key,value in runID_RAM_TLs.items():
            group[str(key).zfill(4)] = np.array(value[:])
        h.close()


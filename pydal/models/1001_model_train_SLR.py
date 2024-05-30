# -*- coding: utf-8 -*-
"""
Do SLR properly, use a class for the model,
and store results to file

Should mostly be based on script400.
"""


import numpy as np
import matplotlib.pyplot as plt

import pydal.utils
import pydal.data_transforms
import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import functions

import pydal.models.SLR_with_transforms as SLR_with_transforms


for YEAR in _vars.YEARS:
    if YEAR == 'All' : 
        fname2019   = r'concatenated_data_2019.pkl'
        fname2020   = r'concatenated_data_2020.pkl'
        data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
        data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
        data        = pydal.utils.concat_dictionaries(data2019,data2020)
    if YEAR == '2019' : 
        fname = r'concatenated_data_2019.pkl'
        data = SLR_with_transforms.load_concat_arrays(fname)
    if YEAR == '2020':
        fname = r'concatenated_data_2020.pkl'
        data = SLR_with_transforms.load_concat_arrays(fname)
         
    #scale the results so behaves well with NN results.   
    rl_s            = data['South'] # 2d array, zero mean gram
    rl_n            = data['North'] # 2d array, zero mean gram
    data['South']   = rl_s / _vars.RL_SCALING #normalize to roughly -1/1    
    data['North']   = rl_n / _vars.RL_SCALING #normalize to roughly -1/1    
    data['X']       = data['X'] / _vars.X_SCALING
    data['Y']       = data['Y'] / _vars.Y_SCALING
        
    for STANDARD in _vars.STANDARDS:

        masked_data = SLR_with_transforms.mask_data(data,STANDARD)
        
        # m,b,r,p,err
        n_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = masked_data['X'],
            p_y             = masked_data['Y'],
            p_theta         = np.zeros_like(masked_data['X']), #not used placeholder
            p_gram          = masked_data['North'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )

        d_n = functions.set_directory_struct(_dirs.DIR_SINGLE_F_SLR,'NORTH')
        f_n =  STANDARD + r'_' + YEAR + '.pkl'
        pydal.utils.dump_pickle_file(
            n_result_db,
            p_data_dir = d_n,
            p_fname     = f_n)

        
        s_result_db     = SLR_with_transforms.SLR_with_y_transform(
            p_x             = masked_data['X'],
            p_y             = masked_data['Y'],
            p_theta         = np.zeros_like(masked_data['X']), #not used placeholder
            p_gram          = masked_data['South'],
            p_x_transform   = pydal.data_transforms.x_transform_y_only,
            # p_x_transform   = pydal.data_transforms.x_transform_x_only,
            p_y_transform   = pydal.data_transforms.no_2d_transform
            )

        d_s = functions.set_directory_struct(_dirs.DIR_SINGLE_F_SLR,'NORTH')
        f_s = STANDARD + r'_' + YEAR + '.pkl'
        pydal.utils.dump_pickle_file(
            s_result_db,
            p_data_dir = d_s,
            p_fname = f_s)
        
        
        
        
        
        
        
        
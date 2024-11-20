# -*- coding: utf-8 -*-
"""

From already-computed propagation model results, calculate the target results.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pydal._directories_and_files as _dirs
import pydal._variables as _vars

root_directory = _dirs.DIR_DATA_PROP_MODELS

FREQS           = np.arange(30,301)
HYDROS          = ['North','South']
MODEL_TYPES     = ['RAM','BELL','KRAK']
COORDINATE      = 'Y'


def calculate_bias(
        x,
        y,
        sl_nom= 160 ):
    """
    This implements an ideal case:
    B = delta = L_{S,New} - L_{S,Old}
    
    recall that tl_var is actually rl-rl_bar, so want to
    subtract not add.
    
    ( in code, delta = rl_db_mean - sl_nom )

    therefore:
    delta > 0   ==> current method produces a higher level than true
    delta < 0   ==> current method produces a lower level than true
    """
    reference_index     = np.argmin(x)    
    tl_var              = yy - np.mean(yy)
    tl_var              = tl_var - tl_var[reference_index]
    
    # now, create what the RL would be while accounting for the 
    # TL variation model.
    rl              = sl_nom - tl_var
    rl_lin          = _vars.REF_UPA * (10 ** ( ( rl / 10 )))
    rl_lin_mean     = np.mean(rl_lin,axis=-1)
    rl_db_mean      = 10*np.log10(rl_lin_mean / _vars.REF_UPA)    
    delta           = rl_db_mean - sl_nom

    return delta


for HYDRO in ['North','South']:
    # HYDRO = 'South'
    directory = root_directory  + r'patricia_bay\\' + HYDRO + '\\' + r'data\\' 

    results_all = dict()
    for FREQ in FREQS:
        results_local = dict()
    
        lines = []
        try:
            with open(directory + r'patbay_'+str(FREQ)+'.txt', 'r') as f:
                lines = f.readlines()
        except:
            # a few frequencies do not work.
            continue
        
        data = dict()
        for line in lines:
            strs        = line.split(':')
            key         = strs[0]
            values      = strs[1].split(',')[:-1]
            data[key]   = values
            
        keys            = list(data.keys())
        for m in MODEL_TYPES:
            working_keys   = [ x for x in keys if m in x] 
            x_key          = [ x for x in working_keys if 'X' in x][0]
            y_key          = [ x for x in working_keys if 'Y' in x][0]
        
            x           = np.array(data[x_key],dtype='complex')
            y           = np.array(data[y_key],dtype='complex')
            index       = x < 141 
            xx          = x[index]
            yy          = y[index]
            yy = np.sqrt ( yy.real ** 2  +  yy.imag ** 2 ) # due to complex array type.
            if 'BELL' in x_key :
                yy = 20*np.log10(yy)
            else:
                yy = -1 * yy
            new_key = m + r'_PL_Bias_' + HYDRO + r'_Y'
            results_local [new_key] = calculate_bias(xx,yy)
        results_all[FREQ] = results_local
        
    df          = pd.DataFrame.from_dict(results_all,orient='index')
    fname       = COORDINATE + '_' + HYDRO + '_dependent_bias_propagation_models.csv'
    df.to_csv(_dirs.DIR_RESULT + fname)

# -*- coding: utf-8 -*-
"""


Why is there a heading dependence in final result??


"""


import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import pydal
import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal._thesis_constants as _thesis

HYDRO               = 'NORTH'
COORD               = 'Y'
N_LAYER             = 1 
N_NODE              = 14
LAYER               = str(N_LAYER) +' layers'
NODE                = str(N_NODE) + ' nodes'

Y_REF               = 0.0

RUN_SELECT_LIST     = [2,3]

FMIN                = 30
FMAX                = 301



"""
1) and 2) GET AND PRUNE X-Y-GRAM DATASET :
"""
fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
# data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
data        = data2020
# data        = pydal.utils.concat_dictionaries(data2019,data2020)

f           = data['Frequency']
rl_s        = data['South'] # 2d array, zero mean gram
rl_n        = data['North'] # 2d array, zero mean gram
# Add 100 to these to get back to non zmrl. 
# Note this will make all speeds look the same!!!!
# (On absolute scales)
rl_s        = rl_s + 100
rl_n        = rl_n + 100
# Note this will make all speeds look the same!!!!

x           = data['X'] / _vars.X_SCALING
y           = data['Y'] / _vars.Y_SCALING
runs        = data['Runs']
run_lengths = data['Run_Lengths']

f_min_ind   = pydal.utils.find_target_freq_index(FMIN, f)
f_max_ind   = pydal.utils.find_target_freq_index(FMAX, f)

frange      = f[f_min_ind:f_max_ind]
rl_sf       = rl_s[f_min_ind:f_max_ind]
rl_nf       = rl_n[f_min_ind:f_max_ind]


"""
3) Load all models in to a dict
"""


nn_dir      = _dirs.DIR_SINGLE_F_1D_NN + r'hdf5_spectrogram_bw_1.0_overlap_90' \
    + r'/' + HYDRO \
    + r'/' + COORD \
    + r'/' + LAYER \
    + r'/' + NODE + r'/'


models_dict = dict()

for freq in frange:
    ff          = int(freq)
    fname       = str(ff).zfill(4) + r'.trch'
    f_model     = nn_dir + fname
    models_dict[ff]  = pydal.models.classes.DeepNetwork_1d(N_LAYER,N_NODE)
    models_dict[ff].load_state_dict(torch.load(f_model))
    models_dict[ff].eval()

L_S_NOW_LIST    = []
L_S_COR_LIST    = []
YYS_LIST        = []
CORRS           = []
for i in range(len(run_lengths)-1): #-1 because of the nan value at end.
# for i in RUN_SELECT_LIST: #-1 because of the nan value at end.
    start       = int(np.sum(run_lengths[:i]))
    end         = int(start + run_lengths[i])
    # plt.scatter(x[start:end],y[start:end])
    yy          = torch.tensor(y[start:end])
    YYS_LIST.append (np.array(yy))
    rl_nn       = rl_nf[:,start:end]    
    rl_ss       = rl_sf[:,start:end] # Not used.
    # Straight up arithmetic average as is done now.
    L_S_NOW     = np.mean(
        _vars.REF_UPA * 10 ** ( rl_nn / 10 ),# put in real domain
        axis=1 #along freq axis
        )
    L_S_NOW_LIST.append ( 10 * np.log10 ( L_S_NOW / _vars.REF_UPA ) )    

    # Now corrected by the learnt N_PL,var
    L_S_COR = np.zeros_like(frange)
    for ii in range(len(frange)):
        corr    = np.zeros_like(yy)
        for iii in range(len(yy)):
            z       = yy[iii]
            z       = z.float()
            z       = z.reshape((1,1))           
            corr[iii] = \
                np.float64(
                    models_dict[frange[ii]].neural_net(z) * _vars.RL_SCALING
                    )
        # Need to refer to CPA (y = 0)
        ref_y       = torch.tensor(Y_REF).float().reshape((1,1))
        val_ref     = models_dict[frange[ii]].neural_net(ref_y) * _vars.RL_SCALING
        corr_zerod  = corr - np.float64(val_ref)
        rr          = rl_nn[ii,:] - corr_zerod  
        CORRS.append(corr)
        L_S_COR[ii]     = np.mean(
            _vars.REF_UPA * 10 ** ( rr / 10 ) # put in real domain
            )
    L_S_COR_LIST.append ( 10 * np.log10 ( L_S_COR / _vars.REF_UPA ) )    


"""


DELTA = []
for n,c in zip(L_S_NOW_LIST,L_S_COR_LIST):
   DELTA.append( c - n )

plt.figure()
DELTA_ARR = np.array(DELTA)
d_mean = np.mean(DELTA_ARR,axis=0)
plt.plot(frange,d_mean,label='DELTA')

plt.figure()
NOW_ARR = np.array(L_S_NOW_LIST)
n_mean = np.mean(NOW_ARR, axis=0)
plt.plot(frange,n_mean,label='NOW')

COR_ARR = np.array(L_S_COR_LIST)
c_mean = np.mean(COR_ARR, axis=0)
plt.plot(frange,c_mean,label='COR')
plt.legend()

for d in DELTA:
    plt.plot(d)
    
plt.figure()
for c,ylab in zip(L_S_COR_LIST,runs):
# for c,ylab in zip(L_S_NOW_LIST,runs):
    if ylab[-2] == 'E':
        plt.plot(c,label=ylab,linestyle='dashed')
    if ylab[-2] == 'W':
        plt.plot(c,label=ylab)
plt.legend()
    

"""



# result          = []
# test            = torch.tensor(features)
# with torch.no_grad():
#     for t in test:
#         t = t.float()
#         t = t.reshape((1,1))
#         result.append(model.neural_net(t))                

# result          = np.array(result) * _vars.RL_SCALING





















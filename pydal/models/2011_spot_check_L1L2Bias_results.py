# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:01:28 2024

@author: Jasper
"""

import torch

import matplotlib.pyplot as plt

import pydal.utils
import pydal.models.functions as functions
import pydal.models.classes as classes
import pydal._variables as _vars
import pydal._directories_and_files as _dirs

fixed_seed  = torch.manual_seed(_vars.SEED) 

hydro = 'North'
coordinate = 'Y'
n_layer = 1
n_node = 26
target_freq = 75
year = '2019'

try :
    type(data)
except:
    data                = \
        pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
    f,rl_s,rl_n,x,y     = \
        data['Frequency'],data['North'],data['South'],data['X'],data['Y']
    run_lengths         = data['Run_Lengths']
    x,y                 = torch.tensor(x),torch.tensor(y)
    
    rl_n = torch.tensor(rl_n * _vars.RL_SCALING)
    rl_s = torch.tensor(rl_s * _vars.RL_SCALING)

if coordinate == 'X':
    feature = x
if coordinate == 'Y':
    feature = y
    
if hydro.capitalize() == 'North':
    rl_true = rl_n
if hydro.capitalize() == 'South':
    rl_true = rl_s
    
freq_index = pydal.utils.find_target_freq_index(target_freq, f)    



# NN result :

# Standard NNs    
# fname   = \
#     functions.set_NN_path(_dirs.DIR_SINGLE_F_1D_NN, hydro, coordinate, n_layer, n_node)
# fname  += str(target_freq).zfill(4)+'.trch'
# model_nn   = classes.DeepNetwork_1d(n_layer,n_node)
# model_nn.load_state_dict(torch.load(fname))    
# high capacity NN
fname = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/NORTH/Y/high capacity//' \
    + str(target_freq).zfill(4) \
    + '.trch'
model_nn   = classes.DeepNetwork_1d(2,512)
model_nn.load_state_dict(torch.load(fname))    

x_nn,y_hat_nn,y_nn= \
    functions.get_predicted_and_true_single_model_percentile(
        p_feature = feature,
        p_rl = rl_true,
        p_model = model_nn,
        p_freq_basis = f,
        p_target_freq = target_freq,
        p_hydro = 'North',
        p_percentile=0)    

# SLR result:
dir_slr         = \
    functions.set_directory_struct(_dirs.DIR_SINGLE_F_SLR,hydro.capitalize())
dir_slr         = dir_slr + coordinate + r'\\'
fname_slr       = r'STANAG_' + year + r'.pkl'
slr_dict        = \
    pydal.utils.load_pickle_file(dir_slr,fname_slr)
slr_m           = slr_dict['m']
slr_b           = slr_dict['b']
model_slr = classes.SLR_1d(
        f[freq_index],
        slr_m[freq_index],
        slr_b[freq_index])
x_slr,y_hat_slr,y_slr= \
    functions.get_predicted_and_true_single_model_percentile(
        p_feature = feature,
        p_rl = rl_true,
        p_model = model_slr,
        p_freq_basis = f,
        p_target_freq = target_freq,
        p_hydro = 'North',
        p_percentile=0)
    
    
plt.figure()
plt.scatter(x_nn,y_nn,marker='.',label='true (nn)')
plt.scatter(x_slr,y_slr,marker='.',label='true (slr)')
plt.scatter(x_nn,y_hat_nn,marker='.',label='nn prediction')
plt.scatter(x_slr,y_hat_slr,marker='.',label='slr prediction')
plt.legend()










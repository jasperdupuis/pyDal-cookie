# -*- coding: utf-8 -*-
"""

functions

"""

import os
import torch

import numpy as np
import matplotlib.pyplot as plt

import pydal.models.classes as classes
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

fixed_seed  = torch.manual_seed(_vars.SEED)

def set_directory_struct(path,p_hydro):
    # Set up the target directory, create if it doesn't exist.
    # The root directory that differentiates based on _variables.py
    if not ( os.path.isdir(path)) : # need to make dir if doesnt exist
        os.mkdir(path)
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    dir_target  = path + dir_spec_subdir # no ending \\ for os.mkdir
    if not ( os.path.isdir(dir_target)) : # need to make parent if doesnt exist
        os.mkdir(dir_target)
    
    # Now teh hydrophone specification:    
    dir_target  = path + dir_spec_subdir + r'\\' + p_hydro 
    if not ( os.path.isdir(dir_target)) : # need to make dir if doesnt exist
        os.mkdir(dir_target)
    dir_target = dir_target + r'\\'
    
    return dir_target

def set_NN_path(parent,hydro,coordinate,layer,node):
    path = set_directory_struct(parent,hydro)
    path = path \
        + coordinate + r'\\' \
        + str(layer) + r' layers\\' \
        + str(node) + r' nodes\\'
    return path

def get_predicted_and_true_single_model_percentile(
        p_feature,
        p_rl,
        p_model,
        p_freq_basis,
        p_target_freq = 50,
        p_hydro = 'North',
        p_percentile=10
        ):

    freq_index = pydal.utils.find_target_freq_index(p_target_freq, p_freq_basis)    
    
    rl_true = p_rl[freq_index,:]
    rl_true = np.array(rl_true)
            
    #Split the data according to how it was trained:
    dset_full               = classes.f_y_orca_dataset(p_feature, rl_true)
    test_size               = int ( len(p_feature) * _vars.TEST)
    hold_size               = int ( len(p_feature) * _vars.HOLDBACK)
    train_size              = len(dset_full) - test_size - hold_size
    _,_,dset_hold    = \
        torch.utils.data.random_split(
            dataset     = dset_full,
            lengths     = [train_size,test_size,hold_size],
            generator   = fixed_seed  )
    #Now the test vectors at this stage can be recovered:                    
    feature_samp    = p_feature[dset_hold.indices]
    label_samp      = rl_true[dset_hold.indices]

    cutoff          = np.percentile(rl_true,p_percentile)
    mask            = label_samp > cutoff
    label_masked    = label_samp[mask]
    feature_masked  = feature_samp[mask]
    
    # initialize result array (absolute results, not residuals)
    temp = np.zeros(len(feature_masked)) 
    temp = torch.tensor(temp)
    for index in range(len(feature_masked)):
        t           = feature_masked[index].float()
        t           = t.reshape((1,1))
        temp[index] = p_model.neural_net(t)
    # if not ('SLR' in key): # scaling is diferernt between slr and nn
    #     # Only the scaled NN's need this correction.                
    #     temp = temp * _vars.RL_SCALING
    temp = temp * _vars.RL_SCALING 
    temp = temp.detach().cpu().numpy()

    return feature_masked,temp,label_masked


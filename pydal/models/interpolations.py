# -*- coding: utf-8 -*-
"""

Interpolations for each model type

These allow y_hat and y_i direct comparisons (estimates vs true values) for
things like L2 error. 

The output(s) should be of the same types for the public functions.

The test domain is PASSED to these functions. i.e., built elsewhere. The test
domain is expected to be NORMALIZED according to values in pydal._variables

"""

import scipy.interpolate as interp
import torch
import numpy as np

import classes
import pydal._variables as _vars

def interpolate_1d_y_SLR(
        p_slopes,
        p_freqs,
        p_domain,
        p_RL_scale = _vars.RL_SCALING,
        p_y_scale = _vars.Y_SCALING):
    """
    
    Apply the slope value along the y-axis values provided.

    Return a 2d array with indices corresponding to frequency and y coordinate
    
    for returned value rrr:
    dimension 0 is domain coordiante (-100 is at 0) 
    and 
    dimension 1 is frequency coordinate ( 2 is at 0)
    
    """
    p_domain    = p_domain * p_y_scale
    p_domain    = np.reshape(p_domain,newshape = (len(p_domain),1))

    result = np.ones((len(p_domain),len(p_freqs)))
    rr = p_domain * result     
    rrr = rr * p_slopes

    return rrr


def estimate_single_1d_y_ML(p_y,p_fname):
    """
    For a single frequency, for a single y-vector, compute the model output.
    
    p_fname is determined outside this function.
    
    The returned result is NOT scaled by _vars.RL_SCALING
    """
    test        = torch.tensor(p_y)
    model       = classes.DeepNetwork()
    model.load_state_dict(torch.load(p_fname))
    model.eval()
    result = np.zeros_like(p_y)
    with torch.no_grad():
        for index in range(len(test)):
            t = test[index].float()
            t = t.reshape((1,1))
            result.append(model.neural_net(t))                
    
    return result # NOT scaled by RL_SCALING!


def interpolate_1d_y_ML(
        p_slopes,
        p_freqs,
        p_domain,
        p_dir, # This is what differentiates north, south, delta DSP, etc
        p_RL_scale = _vars.RL_SCALING,
        p_y_scale = _vars.Y_SCALING):
    """
    
    """
    result = np.ones((len(p_domain),len(p_freqs)))
    
    
    for f in p_freqs:
        fname   = p_dir +  str ( int( f )).zfill(4) + '.trch'        
        temp    = estimate_single_1d_y_ML(p_domain,fname)


    return
    
    
    
    
    





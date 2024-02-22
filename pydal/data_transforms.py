# -*- coding: utf-8 -*-
"""

Repository for fairly simple data operations that could be useful
in various ways (visualizing, modelling, processing, etc) and so doesn't
belong in any subfolder

"""

import numpy as np


def sine_hypothesis_model(x,A,b,phi,c): 
    """
    Fully specified sinusoid 
    """
    y = A * np.sin( (b*x) + phi ) + c 
    return y 


def x_transform_theta_only(x,y,theta):
    """
    x,y,theta are ship position in transformed range coordinates and theta 
    is angle from ship according to standard:
    fwd = 0 stbd = 90, aft = 180 etc
    """
    return theta    


def x_transform_y_only(x,y,theta):
    """
    x,y,theta are ship position in transformed range coordinates and theta 
    is angle from ship according to standard:
    fwd = 0 stbd = 90, aft = 180 etc
    """
    return y   

def x_transform_x_only(x,y,theta):
    """
    x,y,theta are ship position in transformed range coordinates and theta 
    is angle from ship according to standard:
    fwd = 0 stbd = 90, aft = 180 etc
    """
    return x   


def y_transform_0_mean_max_norm_arcsin(p_z):
    """
    Transforms array data using vectorization from passed set to:
        zero mean
        normed on -1,1 using np.max
        arcsin using np.arcsin
    """
    z_means             = np.mean ( p_z , axis = 1)
    interim             = p_z - z_means[:, np.newaxis]
    z_norms             = np.max ( np.abs ( interim ) , axis = 1)
    z_norm_arr          = interim / z_norms[:,np.newaxis]
    z_arcsin_arr        = np.arcsin(z_norm_arr)
    return z_arcsin_arr


def y_transform_0_mean_1d(p_z):
    """
    Transforms 1d data using vectorization from passed set to:
        zero mean
        normed on -1,1 using np.max
        arcsin using np.arcsin
    """
    z_means             = np.mean ( p_z )
    interim             = p_z - z_means
    return interim


def y_transform_0_mean(p_z):
    """
    Transforms array data using vectorization from passed set to:
        zero mean
        normed on -1,1 using np.max
        arcsin using np.arcsin
    """
    z_means             = np.mean ( p_z , axis = 1)
    interim             = p_z - z_means[:, np.newaxis]
    return interim

def no_2d_transform(array):
    """
    Don't do anything to the passed single-argument array
    """
    return array
# -*- coding: utf-8 -*-
"""

Get all results of SLR, ML, and simple averaging and put them in to 
an easily interoperable format.

"""


import numpy as np
import matplotlib.pyplot as plt


import pydal.utils
import pydal.data_transforms
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

import interpolations
import SLR_with_transforms


fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
data2019    = SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = SLR_with_transforms.load_concat_arrays(fname2020)
data        = pydal.utils.concat_dictionaries(data2019,data2020)
data['North'] = data['North'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML,:]
data['South'] = data['South'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML,:]
frequency   = data['Frequency'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML]
del data2019 # Not needed anymore, free memory
del data2020 # Not needed anymore, free memory


# Set up the test domain.
# These are normalized values!
xmin = -1
xmax = 1
ymin = -1
ymax = 1
# Setup the x vector, y vector, and 2d space.
x_range     = np.arange(xmin,xmax,step=0.01)
y_range     = np.arange(ymin,ymax,step=0.01)


# # 1
#
# Calculate the SLR results (this is fast, can be redone from data):
# Stores to dictionaries with keys m,b,r,p,err across frequency range
slopes_db_n     = SLR_with_transforms.SLR_with_y_transform(
    p_x             = data['X'] / _vars.X_SCALING,
    p_y             = data['Y'] / _vars.Y_SCALING,
    p_theta         = np.zeros_like(data['X']), #not used placeholder
    p_gram          = data['North'] / _vars.RL_SCALING,
    p_x_transform   = pydal.data_transforms.x_transform_y_only,
    # p_x_transform   = pydal.data_transforms.x_transform_x_only,
    p_y_transform   = pydal.data_transforms.no_2d_transform
    )

slopes_db_s     = SLR_with_transforms.SLR_with_y_transform(
    p_x             = data['X'] / _vars.X_SCALING,
    p_y             = data['Y'] / _vars.Y_SCALING,
    p_theta         = np.zeros_like(data['X']), #not used placeholder
    p_gram          = data['South'] / _vars.RL_SCALING,
    p_x_transform   = pydal.data_transforms.x_transform_y_only,
    # p_x_transform   = pydal.data_transforms.x_transform_x_only,
    p_y_transform   = pydal.data_transforms.no_2d_transform
    )

result_SLR_s   = interpolations.interpolate_1d_y_SLR(
    slopes_db_s['m'], frequency, y_range)
result_SLR_n   = interpolations.interpolate_1d_y_SLR(
    slopes_db_n['m'], frequency, y_range)


# # 2
#
# ML results
# Y-coordinate feature only

dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
    _vars.FS_HYD,
    _vars.T_HYD_WINDOW * _vars.FS_HYD,
    _vars.OVERLAP
    )
dir_target_s    = _dirs.DIR_SINGLE_F_NN \
                + dir_spec_subdir \
                + r'\\South\\'
dir_target_n    = _dirs.DIR_SINGLE_F_NN \
                + dir_spec_subdir \
                + r'\\North\\'

result_ML_y_1d_n = interpolations.interpolate_1d_y_ML(
    frequency, y_range, dir_target_n)
result_ML_y_1d_s = interpolations.interpolate_1d_y_ML(
    frequency, y_range, dir_target_s)


# Compare two models 
p_model_1 = result_ML_y_1d_s                #2d RL variation as f(position, frequency)
# p_model_2 = np.zeros_like(result_ML_y_1d_n) #2d RL variation as f(position, frequency)
p_model_2 = result_SLR_s
p_fs = frequency
p_sl_nom = 100

rl_1            = p_sl_nom + p_model_1
rl_1_lin        = _vars.REF_UPA * (10 ** ( ( rl_1 / 10 )))
rl_1_lin_mean     = np.mean(rl_1_lin,axis=0)
rl_1_db_mean      = 10*np.log10(rl_1_lin_mean / _vars.REF_UPA)

rl_2            = p_sl_nom + p_model_2
rl_2_lin        = _vars.REF_UPA * (10 ** ( ( rl_2 / 10 )))
rl_2_lin_mean     = np.mean(rl_2_lin,axis=0)
rl_2_db_mean      = 10*np.log10(rl_2_lin_mean / _vars.REF_UPA)


plt.figure();plt.plot(p_fs,rl_1_db_mean-p_sl_nom)
plt.figure();plt.plot(p_fs,rl_2_db_mean-p_sl_nom)









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

import SLR_with_transforms


fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
data2019    = SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = SLR_with_transforms.load_concat_arrays(fname2020)
data        = pydal.utils.concat_dictionaries(data2019,data2020)
data['North'] = data['North'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML,:]
data['South'] = data['South'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML,:]

frequency   = data['Frequency'][_vars.MIN_F_INDEX_ML:_vars.MAX_F_INDEX_ML]

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
n_result_db     = SLR_with_transforms.SLR_with_y_transform(
    p_x             = data['X'] / _vars.X_SCALING,
    p_y             = data['Y'] / _vars.Y_SCALING,
    p_theta         = np.zeros_like(data['X']), #not used placeholder
    p_gram          = data['North'] / _vars.RL_SCALING,
    p_x_transform   = pydal.data_transforms.x_transform_y_only,
    # p_x_transform   = pydal.data_transforms.x_transform_x_only,
    p_y_transform   = pydal.data_transforms.no_2d_transform
    )

s_result_db     = SLR_with_transforms.SLR_with_y_transform(
    p_x             = data['X'] / _vars.X_SCALING,
    p_y             = data['Y'] / _vars.Y_SCALING,
    p_theta         = np.zeros_like(data['X']), #not used placeholder
    p_gram          = data['South'] / _vars.RL_SCALING,
    p_x_transform   = pydal.data_transforms.x_transform_y_only,
    # p_x_transform   = pydal.data_transforms.x_transform_x_only,
    p_y_transform   = pydal.data_transforms.no_2d_transform
    )


# # 2
#
# ML results
# Y-coordinate feature only

p_track_dist_m  = 200
p_track_step    = 2
p_m_values      = n_result_db['m']

track_steps     = np.arange(p_track_dist_m / p_track_step) 
track_steps     -= (len(track_steps) // 2 )
track_steps     *= p_track_step
track_steps     = np.reshape(track_steps,( len ( track_steps) , 1 ) )
slope_per_100m  = np.reshape(p_m_values, ( 1 , len ( p_m_values)))

tl_var          = np.multiply(track_steps,slope_per_100m)

# now, create what the RL would be while accounting for the 
# linear TL variation model.
rl              = p_sl_nom + tl_var
rl_lin          = _vars.REF_UPA * (10 ** ( ( rl / 10 )))
rl_lin_mean     = np.mean(rl_lin,axis=0)
rl_db_mean      = 10*np.log10(rl_lin_mean / _vars.REF_UPA)


test            = torch.tensor(y_range)
with torch.no_grad():
    for t in test:
        t = t.float()
        t = t.reshape((1,1))
        result.append(model.neural_net(t))                
result = np.array(result) * _vars.RL_SCALING


import scipy.interpolate as interp





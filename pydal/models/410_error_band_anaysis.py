# -*- coding: utf-8 -*-
"""

Quantify the distribution of the zero mean spectral series. 

Do this by zero-meaning on narrow criteria, namely small dy segments
(default is 5m wide bins), which will avoid TL_y sensitivity but 
should leave only RV suitable data.



"""

import matplotlib.pyplot as plt
import numpy as np

import pydal.utils
import pydal.data_transforms
import pydal.models.SLR_with_transforms

import pydal._variables as _vars
import pydal._directories_and_files as _dirs


# load zero-mean data as used for SLR
# note function loads 2019 by default, 2020 is provided hard stringed.
fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
data        = pydal.utils.concat_dictionaries(data2019,data2020)

f = data['Frequency']

# Determine the STD of each zero-mean chunk on the y-axis
n_y_sets    = _vars.N_Y_BINS               # N segments there are
y_step      = _vars.Y_LENGTH  / n_y_sets   # how long each seg is
y_values    = ( np.arange( n_y_sets + 1 ) * y_step ) - 100  # array values from above
# results store in these lists
s_std = []
n_std = []
for index in range(len(y_values)):
    #    index = 2      # value for testing loop
    # Build a mask to get the right Y and label data    
    mask_low    = y_values[index] < data['Y'] 
    mask_high   = data['Y'] < (y_values[index] + y_step)
    mask        = np.logical_and(mask_low,mask_high)
    ss          = data['South'][:,mask]    
    nn          = data['North'][:,mask]
    # compute the zero mean arrays
    ss          = pydal.data_transforms.y_transform_0_mean(ss)
    nn          = pydal.data_transforms.y_transform_0_mean(nn)
    # compute and append the std of the zero-meaned sets
    s_std.append(np.std(ss,axis=1))
    n_std.append(np.std(nn,axis=1))

fig,ax = plt.subplots()
for index in range (len(y_values))    :
    ax.plot(f[:10000],s_std[index],label=str(y_values[index]))
ax.set_xscale('log')
fig.legend()
    
# histogram for a specific y_length index, 10 is ~CPA
index = 15
mask_low    = y_values[index] < data['Y'] 
mask_high   = data['Y'] < (y_values[index] + y_step)
mask        = np.logical_and(mask_low,mask_high)
ss          = data['South'][:,mask]    
nn          = data['North'][:,mask]
ss          = pydal.data_transforms.y_transform_0_mean(ss)
nn          = pydal.data_transforms.y_transform_0_mean(nn)
plt.figure();plt.hist(nn.flatten(),bins=50)
plt.title(str(y_values[index]))    


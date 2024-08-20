# -*- coding: utf-8 -*-
"""


"""

import matplotlib.pyplot as plt

import numpy as np

import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

import pydal.models.SLR_with_transforms
import scipy.stats as stats

fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
data_all    = pydal.utils.concat_dictionaries(data2019,data2020)


#
#
# 2019 DATA BELOW

data = data2019
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['North'],data['South'],data['X'],data['Y']
run_lengths         = data['Run_Lengths']    
freq_targ = 73
freq_ind  = pydal.utils.find_target_freq_index(freq_targ, f)

# Mask according to percentile
mask_hi  = x > np.percentile(x,90)
mask_lo  = x < np.percentile(x,10)

n_hi = rl_n[freq_ind  ,mask_hi]
n_lo = rl_n[freq_ind  ,mask_lo]

reg2019 = stats.linregress(x,rl_n[freq_ind,:])
print(reg2019)

plt.figure('2019_scatter_x_only')
start = 0
for n in run_lengths:
    # plt.scatter(x[start:start + n],y[start:start + n])
    plt.scatter(
        x[start:start + n],
        rl_n[freq_ind,start:start + n],
        marker='.')
    start = start + n


#
#
# 2020 DATA BELOW

data = data2020
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['North'],data['South'],data['X'],data['Y']
run_lengths         = data['Run_Lengths'] 
   
mask_hi  = x > np.percentile(x,90)
mask_lo  = x < np.percentile(x,10)

reg2020 = stats.linregress(x,rl_n[freq_ind,:])
m = reg2020[0]
print(reg2020)


plt.figure('2020_scatter_y_with_x_percentiles') 
plt.scatter(y,rl_n[freq_ind,:],marker='.')
plt.scatter(y[mask_hi],rl_n[freq_ind,mask_hi],marker='.')
plt.scatter(y[mask_lo],rl_n[freq_ind,mask_lo],marker='.')


plt.figure('2020_scatter_x_only')
start = 0
for n in run_lengths:
    # plt.scatter(x[start:start + n],y[start:start + n])
    plt.scatter(
        # np.arange(n),
        y[start:start + n],
        rl_n[freq_ind,start:start + n],
        marker='.')
    start = start + n
    

mid_only = np.logical_or(mask_hi,mask_lo)
mid_only = np.logical_not(mid_only)

xx = x[mid_only]
nn = rl_n[freq_ind,mid_only]
plt.figure('2020_scatter_x_only_middle_80')
plt.scatter(xx,nn,marker='.')

import torch
import pydal.models.classes
name_x       = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/NORTH/X/1 layers/14 nodes/0073.trch'
name_y       = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/NORTH/Y/1 layers/14 nodes/0073.trch'

modelx       = pydal.models.classes.DeepNetwork_1d(1,14)
modelx.load_state_dict(torch.load(name_x))
modelx.eval()

modely       = pydal.models.classes.DeepNetwork_1d(1,14)
modely.load_state_dict(torch.load(name_y))
modely.eval()

features    = np.arange(-1,1,step=0.01)
test            = torch.tensor(features)
resultx = []
resulty = []
with torch.no_grad():
    for t in test:
        t = t.float()
        t = t.reshape((1,1))
        resultx.append(modelx.neural_net(t))    
        resulty.append(modely.neural_net(t))    
resultx = np.array(resultx)
resulty = np.array(resulty)

plt.figure('this one');
plt.scatter(x,rl_n[freq_ind,:],marker='.')
plt.scatter(features * _vars.X_SCALING,resultx * _vars.RL_SCALING,marker='.')
plt.scatter(features* _vars.X_SCALING , features*m*_vars.RL_SCALING,marker='.')

plt.figure();
plt.scatter(y/_vars.Y_SCALING,rl_n[freq_ind,:]/_vars.RL_SCALING,marker='.')
plt.scatter(features,resulty,marker='.')


#
#
# Look cross frequency, maybe this will make it easier to see something...?

data = data_all
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['North'],data['South'],data['X'],data['Y']
run_lengths         = data['Run_Lengths'] 


freqs       = np.arange(40,100)
f_lo_ind    = pydal.utils.find_target_freq_index(min(freqs), f)
f_hi_ind    = pydal.utils.find_target_freq_index(max(freqs), f)
xhi         = x < np.percentile(x,60)
xlo         = x > np.percentile(x,40)
xmid        = np.logical_and(xhi,xlo)

yhi         = y < np.percentile(y,60)
ylo         = y > np.percentile(y,40)
ymid        = np.logical_and(yhi,ylo)


nn          = rl_n [f_lo_ind:f_hi_ind,xmid]
xx          = x[xmid]

nn          = rl_n [f_lo_ind:f_hi_ind,ymid]
yy          = y[ymid]

plt.figure()
for i in range(len(freqs)):
    plt.scatter(xx,nn[i,:],marker='.')




    

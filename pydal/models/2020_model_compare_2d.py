# -*- coding: utf-8 -*-
"""

The 2D training doesn't work well. Here's why.


"""

import torch
from torch.utils.data import DataLoader
import numpy as np
# import sys
# sys.setrecursionlimit(10000) #default is 3000. Doesnt help my issues.

import classes # local to this directory
import functions # local to this directory
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

nn_2d_layers    = [6]
nn_2d_nodes     = [38]
hydro           = 'NORTH'
target_freq     = 53
standard        = 'ISO'
coordinate      = 'Y'
y_basis         = np.arange(-1,1,0.01)

fixed_seed  = torch.manual_seed(_vars.SEED)
torch.no_grad()

# Real data results:
# rl_n, rl_s, x, y are already scaled
data                = \
    pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['North'],data['South'],data['X'],data['Y']
run_lengths         = data['Run_Lengths']
x,y                 = torch.tensor(x),torch.tensor(y)

freq_index = pydal.utils.find_target_freq_index(target_freq, f)


# make the 2D NN layer models
models = []
names = []
for layer in nn_2d_layers:
    for node in nn_2d_nodes:
        fname   = \
            functions.set_NN_path(_dirs.DIR_SINGLE_F_2D_NN, hydro, layer, node)
        fname  += str(target_freq).zfill(4)+'.trch'
        model   = classes.DeepNetwork_2d(layer,node)
        model.load_state_dict(torch.load(fname))
        # model.eval()
        models.append(model)
        names.append('NN_2d_' + str(layer) + '_layers_' + str(node) + '_nodes')

# Find L1 and L2 for the NN2D models.

if hydro.capitalize() == 'North':
    rl_true = rl_n * _vars.RL_SCALING
if hydro.capitalize() == 'South':
    rl_true = rl_s * _vars.RL_SCALING
rl_true = rl_true[freq_index,:]

# 2D NN DEMONSTRATION SCRIPT
temp = torch.tensor(np.zeros_like(x))
key = names[2]
model = models[2]
for index in range(len(x)):
    t           = y[index],x[index]
    t           = torch.tensor(t).float()
    temp[index] = model.neural_net(t)

L1                  = pydal.utils.L1_error(
    rl_true, 
    temp.detach().cpu().numpy())
new_key = key + r'_L1'
results[new_key] = L1        

L2                  = pydal.utils.L2_error(
    rl_true, 
    temp.detach().cpu().numpy())
new_key = key + r'_L2'
results[new_key] = L2  




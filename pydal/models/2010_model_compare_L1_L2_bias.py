"""

Make a file that takes as input the raw xy and rl data, and then does 
L1 and L2 error for the real track vs model prediction.

arguments : 
    f
    model

inputs:
    x (data)
    y (data)
    f (data)
    rl (data)
    rl_hat (model)
outputs:
    L1
    L2

Operations required:

    0. Select hydrophone and model selection parameters
    1. Get the real data
    2. Instantiate the desired models
    3. Compute L1, L2
    4. Compare outputs (numeric or graphical?)

Each model-freq pair will have its own L1, L2 result

Remember SLR was in real numbers, while NNs were scaled for training.

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


def calculate_bias(basis,
         model,
         sl_nom= 160 / _vars.RL_SCALING):
    """
    This is the bias calcualtion vs straight averaging,
    i.e. violation of stationarity assumption.

    rl_db_mean is the current output, which averages blindly
    sl_nom is what the output of a perfect system should produce

    delta = rl_db_mean - sl_nom

    therefore:
    delta > 0   ==> current method produces a higher level than true
    delta < 0   ==> current method produces a lower level than true
    """
    tl_var = np.zeros_like(basis)
    for index in range(len(basis)):
        t               = basis[index].float()
        t               = t.reshape((1,1))
        tl_var[index]   = model.neural_net(t)
    
    # now, create what the RL would be while accounting for the 
    # linear TL variation model.
    rl              = sl_nom - tl_var
    rl_lin          = _vars.REF_UPA * (10 ** ( ( rl / 10 )))
    rl_lin_mean     = np.mean(rl_lin,axis=0)
    rl_db_mean      = 10*np.log10(rl_lin_mean / _vars.REF_UPA)    
    delta       = rl_db_mean - sl_nom

    return delta  

def calculate_predicted_values_and_errors(model_dictionary,rl_true):
    """
    Note bias is done in the 1d selection if-else.
    
    model_dictionary = model_dict
    rl_true = rl_n
    """
    results = dict()
    for key,model in model_dictionary.items():
        # initialize result array
        temp = np.zeros(len(x))
        temp = torch.tensor(temp)
        # 2d needs x and y features.
        if '2d' in key:
            for index in range(len(x)):
                t           = y[index],x[index]
                t           = torch.tensor(t).float()
                temp[index] = model.neural_net(t)
        # 1d only needs y features
        else:
            for index in range(len(x)):
                t           = y[index].float()
                t           = t.reshape((1,1))
                temp[index] = model.neural_net(t)
            new_key = key + r'_bias'
            results[new_key] = calculate_bias(
                torch.tensor(np.arange(-1,1,0.01)),
                model)
        
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
        
    return results


# 0. Hydrophone and model selection parameters
hydro           = 'NORTH'
target_freq     = 53
standard        = 'ISO'
y_basis         = np.arange(-1,1,0.01)

nn_1d_layers    = [1]
nn_1d_nodes     = [14]
nn_2d_layers    = [6]
nn_2d_nodes     = [38]
slr_year        = '2020'
sl_nom          = 160

fixed_seed  = torch.manual_seed(_vars.SEED)
torch.no_grad()

#1. Get the real data
"""
len(f) is 84,998.
Elsewhere, it was truncated using f[:max], so can be searched for
low values interchangeably.

In short: can use freq_index for both frequency arrays that appear in the work of
length 10000 or 85000. As long as below 9kHz overall.
"""

# rl_n, rl_s, x, y are already scaled
data                = \
    pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['North'],data['South'],data['X'],data['Y']
run_lengths         = data['Run_Lengths']
x,y                 = torch.tensor(x),torch.tensor(y)

freq_index = pydal.utils.find_target_freq_index(target_freq, f)

# 2. Instantiate desired models
"""
1 model per frequency per type of model.
NNs also have layer / node parameters

Always include SLR, otherwise operate from lists of values for NNs
"""

models          = []
names           = []

dir_slr         = \
    functions.set_directory_struct(_dirs.DIR_SINGLE_F_SLR,hydro)
fname_slr       = standard + r'_' + slr_year + r'.pkl'
slr_dict        = \
    pydal.utils.load_pickle_file(dir_slr,fname_slr)
slr_m           = slr_dict['m']
models.append(
    classes.SLR_1d(
        f[freq_index],
        slr_m[freq_index]))
names.append('SLR')
for layer in nn_1d_layers:
    for node in nn_1d_nodes:
        fname   = \
            functions.set_NN_path(_dirs.DIR_SINGLE_F_1D_NN, hydro, layer, node)
        fname  += str(target_freq).zfill(4)+'.trch'
        model   = classes.DeepNetwork_1d(layer,node)
        model.load_state_dict(torch.load(fname))
        # model.eval()
        models.append(model)
        names.append('NN_1d_' + str(layer) + '_layers_' + str(node) + '_nodes')
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

model_dict      = dict(zip(names,models))

# 3. Calculate predicted values and L1, L2 errors and bias for 1d models.
# result_dict  = calculate_predicted_values_and_errors(model_dict, rl_n)

results = dict()

# key = names[1]
# model = models[1]

# # initialize result array
# temp = np.zeros(len(x))
# temp = torch.tensor(temp)
# # 2d needs x and y features.
# if '2d' in key:
#     for index in range(len(x)):
#         t           = y[index],x[index]
#         t           = torch.tensor(t).float()
#         temp[index] = model.neural_net(t)
# # 1d only needs y features
# else:
#     for index in range(len(x)):
#         t           = y[index].float()
#         t           = t.reshape((1,1))
#         temp[index] = model.neural_net(t)
#     new_key = key + r'_bias'
#     results[new_key] = calculate_bias(
#         torch.tensor(np.arange(-1,1,0.01)),
#         model)

# L1                  = pydal.utils.L1_error(
#     rl_n, 
#     temp.detach().cpu().numpy())
# new_key = key + r'_L1'
# results[new_key] = L1        

# L2                  = pydal.utils.L2_error(
#     rl_n, 
#     temp.detach().cpu().numpy())
# new_key = key + r'_L2'
# results[new_key] = L2      

# testing values only
# key = names[0]
# model = models[0]
for key,model in model_dict.items():
    # initialize result array
    temp = np.zeros(len(x))
    temp = torch.tensor(temp)
    # 2d needs x and y features.
    if '2d' in key:
        z=1
        # for index in range(len(x)):
            # t           = y[index],x[index]
            # t           = torch.tensor(t).float()
            # temp[index] = model.neural_net(t)
    # 1d only needs y features
    else:
        for index in range(len(x)):
            t           = y[index].float()
            t           = t.reshape((1,1))
            temp[index] = model.neural_net(t)
        new_key = key + r'_bias'
        results[new_key] = calculate_bias(
            torch.tensor(np.arange(-1,1,0.01)),
            model)
    
    L1                  = pydal.utils.L1_error(
        rl_n, 
        temp.detach().cpu().numpy())
    new_key = key + r'_L1'
    results[new_key] = L1        

    L2                  = pydal.utils.L2_error(
        rl_n, 
        temp.detach().cpu().numpy())
    new_key = key + r'_L2'
    results[new_key] = L2     
    del(temp)
















import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
# import sys
# sys.setrecursionlimit(10000) #default is 3000. Doesnt help my issues.

import classes # local to this directory
import functions # local to this directory
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

import matplotlib.pyplot as plt

PERCENTILE          = 1
ALL_DATA            = False
UPPER_PERCENTILE    = False
RESIDUALS_ONLY      = True
# for single freq retrieval and testing to find rmse and mae issue:
TROUBLESHOOT        = False


# 0. Hydrophone and model selection parameters
p_hydros        = _vars.HYDROS
# p_hydros        = ['SOUTH']
target_freqs    = np.arange(10,301) #10,300 is target.
# target_freqs    = np.arange(88,89) # TESTING LINE
standard        = 'STANAG'
coordinate      = 'Y'

nn_1d_layers    = [1]
nn_1d_nodes     = _vars.LIST_N_NODES_1D
# nn_1d_layers    = []
# nn_1d_nodes     = []
slr_years       = ['2019','2020','All']
sl_nom          = 160

fixed_seed  = torch.manual_seed(_vars.SEED)
torch.no_grad()


def calculate_bias(basis,
         model,
         ref_point=torch.tensor(0.0),
         sl_nom= 160 ):
    """
    This implements an ideal case:
    B = delta = L_{S,New} - L_{S,Old}
    
    recall that tl_var is actually rl-rl_bar, so want to
    subtract not add.
    
    ( in code, delta = rl_db_mean - sl_nom )

    therefore:
    delta > 0   ==> current method produces a higher level than true
    delta < 0   ==> current method produces a lower level than true
    """
    tl_var = torch.zeros_like(basis)
    for index in range(len(basis)):
        t               = basis[index].float()
        t               = t.reshape((1,1))
        tl_var[index]   = model.neural_net(t)
    if 'SLR' in str(type(model)):
        reference = model.b
    else:
        reference = model.neural_net(ref_point.reshape((1,1)))        
    tl_var = tl_var - reference
    tl_var *= _vars.RL_SCALING    

    # now, create what the RL would be while accounting for the 
    # linear TL variation model.
    rl              = sl_nom - tl_var
    rl_lin          = _vars.REF_UPA * (10 ** ( ( rl / 10 )))
    rl_lin_mean     = torch.mean(rl_lin,axis=-1)
    rl_db_mean      = 10*torch.log10(rl_lin_mean / _vars.REF_UPA)    
    delta           = rl_db_mean - sl_nom

    return delta.detach().numpy().item()



fname_target = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/NORTH/Y/high capacity/0200.trch'
model   = classes.DeepNetwork_1d(2,512)
model.load_state_dict(torch.load(fname_target))
y  = ( np.arange(41) * 5 - 100)  / 100
yy = torch.tensor(y)


b = calculate_bias(yy,model)



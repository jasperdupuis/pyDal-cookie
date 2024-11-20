"""

Same as script 2010 except added the high capacity NN for the north hydro only

(Difference appears in get_models() in this file)

"""

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
ALL_DATA            = True
UPPER_PERCENTILE    = False
RESIDUALS_ONLY      = False
# for single freq retrieval and testing to find rmse and mae issue:
TROUBLESHOOT        = False


# 0. Hydrophone and model selection parameters
p_hydros        = _vars.HYDROS
# p_hydros        = ['SOUTH']
# target_freqs    = np.arange(10,301) #10,300 is target.
target_freqs     = [200]
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


def get_models(p_freq,p_freq_index,p_coordinate,p_standard ,p_hydro,p_slr_years,p_layers,p_nodes):
    """
    # 2. Instantiate desired models
    1 model per frequency per type of model.
    NNs also have layer / node parameters

    Always include SLR, otherwise operate from lists of values for NNs
    """
    models          = []
    names           = []        
    dir_slr         = \
        functions.set_directory_struct(_dirs.DIR_SINGLE_F_SLR,p_hydro.capitalize())
    dir_slr         = dir_slr + p_coordinate + r'\\'
    for year in p_slr_years:
        fname_slr       = p_standard + r'_' + year + r'.pkl'
        try:
            slr_dict        = \
                pydal.utils.load_pickle_file(dir_slr,fname_slr)
        except : 
            continue
        slr_m           = slr_dict['m']
        slr_b           = slr_dict['b']
        models.append(
            classes.SLR_1d(
                f[p_freq_index],
                slr_m[p_freq_index],
                slr_b[p_freq_index]))
        names.append('SLR_' + year)    
    
    for layer in p_layers:
        for node in p_nodes:
            fname   = \
                functions.set_NN_path(_dirs.DIR_SINGLE_F_1D_NN, hydro, p_coordinate, layer, node)
            fname  += str(p_freq).zfill(4)+'.trch'
            model   = classes.DeepNetwork_1d(layer,node)
            try:
                model.load_state_dict(torch.load(fname))
            except:
                continue
            # model.eval()
            models.append(model)
            names.append('NN_1d_' + str(layer) + '_layers_' + str(node) + '_nodes')    
    
    
    if p_coordinate == 'Y':
        fname = \
            r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/'+p_hydro+'/Y/high capacity//' \
            + str(p_freq).zfill(4) \
            + '.trch'
        model_nn   = classes.DeepNetwork_1d(2,512)
        model_nn.load_state_dict(torch.load(fname))    
        models.append(model_nn)
        names.append('NN_high_capacity')
        
    # 3. Calculate predicted values and L1, L2 errors and bias for 1d models.
    models_dict  = dict(zip(names,models))
    return models_dict

#1. Get the real data
"""
len(f) is 84,998.
Elsewhere, it was truncated using f[:max], so can be searched for
low values interchangeably.

In short: can use freq_index for both frequency arrays that appear in the work of
length 10000 or 85000. As long as below 9kHz overall.
"""

if __name__ == '__main__':
    # rl_n, rl_s, x, y are scaled by the _vars constants in this call.
    data                = \
        pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
    f,rl_s,rl_n,x,y     = \
        data['Frequency'],data['South'],data['North'],data['X'],data['Y']
    run_lengths         = data['Run_Lengths']
    x,y                 = torch.tensor(x),torch.tensor(y)
    
    rl_n = torch.tensor(rl_n * _vars.RL_SCALING)
    rl_s = torch.tensor(rl_s * _vars.RL_SCALING)

    
    if coordinate == 'X':
        feature = x
    if coordinate == 'Y':
        feature = y
        
    models_dict = dict() # models are here. out of scope for later troubleshooting.
    results_all = dict()
   
    if ALL_DATA:
         
        for freq in target_freqs:
            print(str(freq))
            freq_index = pydal.utils.find_target_freq_index(freq, f)    
            results_local = dict()
            
            for hydro in p_hydros: 
                hydro = hydro.capitalize()
                if hydro.capitalize() == 'North':
                    rl_true = rl_n 
                if hydro.capitalize() == 'South':
                    rl_true = rl_s 
                rl_true = rl_true[freq_index,:]
                rl_true = np.array(rl_true)
                #Split the data according to how it was trained:
                dset_full               = classes.f_y_orca_dataset(feature, rl_true)
                test_size               = int ( len(feature) * _vars.TEST)
                hold_size               = int ( len(feature) * _vars.HOLDBACK)
                train_size              = len(dset_full) - test_size - hold_size
                _,_,dset_hold    = \
                    torch.utils.data.random_split(
                        dataset     = dset_full,
                        lengths     = [train_size,test_size,hold_size],
                        generator   = fixed_seed  )
                #Now the test vectors at this stage can be recovered:                    
                feature_samp    = feature[dset_hold.indices]
                label_samp      = rl_true[dset_hold.indices]
       
                cutoff          = np.percentile(rl_true,10)
                mask            = label_samp > cutoff
                label_masked    = label_samp[mask]
                feature_masked  = feature_samp[mask]
       
                # 2. Instantiate desired models    
                models_dict = get_models(freq,freq_index,coordinate,standard,hydro,slr_years,nn_1d_layers,nn_1d_nodes)
                
                """
                testing:
                keys = list(models_dict.keys())
                key = keys[3]
                model = models_dict[key]
                """
                for key,model in models_dict.items():
                    # initialize result array (absolute results, not residuals)
                    temp = np.zeros(len(feature_masked)) 
                    temp = torch.tensor(temp)
                    for index in range(len(feature_masked)):
                        t           = feature_masked[index].float()
                        t           = t.reshape((1,1))
                        temp[index] = model.neural_net(t)
                    # if not ('SLR' in key): # scaling is diferernt between slr and nn
                    #     # Only the scaled NN's need this correction.                
                    #     temp = temp * _vars.RL_SCALING
                    temp = temp * _vars.RL_SCALING 
                    temp = temp.detach().cpu().numpy()
        
                    new_key = key + r'_Bias_' + hydro + r'_' + coordinate
                    results_local [new_key] = calculate_bias(
                        torch.tensor(np.arange(-1.,1.,0.01)),
                        model)
                    
                    # Add the b-intercept term for MAE and MSE
                    # (This is done internally to the calculate_bias function.)
                    if 'SLR' in key:
                        temp  = temp + model.b
                    
                    L1,MAE                  = pydal.utils.L1_error(
                        label_masked, 
                        temp)
                    new_key = key + r'_L1_' + hydro+ r'_' + coordinate
                    results_local [new_key] = L1        
                    new_key = key + r'_MAE_' + hydro+ r'_' + coordinate
                    results_local [new_key] = MAE        
                
                    L2,MSE                  = pydal.utils.L2_error(
                        label_masked, 
                        temp)
                    new_key = key + r'_L2_' + hydro + r'_' + coordinate
                    results_local [new_key] = L2      
                    new_key = key + r'_MSE_' + hydro + r'_' + coordinate
                    results_local [new_key] = MSE     
                    
            results_all[freq] = results_local
                                
        df          = pd.DataFrame.from_dict(results_all,orient='index')
        fname       = coordinate + '_dependent_L1_L2_bias_high_capacity.csv'
        df.to_csv(_dirs.DIR_RESULT + fname)


    if UPPER_PERCENTILE:

        for freq in target_freqs:
            print(str(freq))
            freq_index = pydal.utils.find_target_freq_index(freq, f)    
            results_local = dict()
            
            for hydro in p_hydros: 
                hydro = hydro.capitalize()
                if hydro.capitalize() == 'North':
                    rl_true = rl_n 
                if hydro.capitalize() == 'South':
                    rl_true = rl_s 
                rl_true = rl_true[freq_index,:]
                rl_true = np.array(rl_true)
                    
                #Split the data according to how it was trained:
                dset_full               = classes.f_y_orca_dataset(feature, rl_true)
                test_size               = int ( len(feature) * _vars.TEST)
                hold_size               = int ( len(feature) * _vars.HOLDBACK)
                train_size              = len(dset_full) - test_size - hold_size
                _,_,dset_hold    = \
                    torch.utils.data.random_split(
                        dataset     = dset_full,
                        lengths     = [train_size,test_size,hold_size],
                        generator   = fixed_seed  )
                #Now the test vectors at this stage can be recovered:                    
                feature_samp    = feature[dset_hold.indices]
                label_samp      = rl_true[dset_hold.indices]

                cutoff          = np.percentile(rl_true,PERCENTILE)
                mask            = label_samp > cutoff
                label_masked    = label_samp[mask]
                feature_masked  = feature_samp[mask]

                # 2. Instantiate desired models    
                models_dict = get_models(freq,freq_index,coordinate,standard,hydro,slr_years,nn_1d_layers,nn_1d_nodes)
                
                """
                testing:
                keys = list(models_dict.keys())
                key = keys[3]
                model = models_dict[key]
                """
                for key,model in models_dict.items():
                    # initialize result array (absolute results, not residuals)
                    temp = np.zeros(len(feature_masked)) 
                    temp = torch.tensor(temp)
                    for index in range(len(feature_masked)):
                        t           = feature_masked[index].float()
                        t           = t.reshape((1,1))
                        temp[index] = model.neural_net(t)
                    # if not ('SLR' in key): # scaling is diferernt between slr and nn
                    #     # Only the scaled NN's need this correction.                
                    #     temp = temp * _vars.RL_SCALING
                    temp = temp * _vars.RL_SCALING 
                    temp = temp.detach().cpu().numpy()
        
                    new_key = key + r'_Bias_' + hydro + r'_' + coordinate
                    results_local [new_key] = calculate_bias(
                        torch.tensor(np.arange(-1.,1.,0.01)),
                        model)
                    
                    if 'SLR' in key:
                        temp  = temp + model.b
                    
                    L1,MAE                  = pydal.utils.L1_error(
                        label_masked, 
                        temp)
                    new_key = key + r'_L1_' + hydro+ r'_' + coordinate
                    results_local [new_key] = L1        
                    new_key = key + r'_MAE_' + hydro+ r'_' + coordinate
                    results_local [new_key] = MAE        
                
                    L2,MSE                  = pydal.utils.L2_error(
                        label_masked, 
                        temp)
                    new_key = key + r'_L2_' + hydro + r'_' + coordinate
                    results_local [new_key] = L2      
                    new_key = key + r'_MSE_' + hydro + r'_' + coordinate
                    results_local [new_key] = MSE      
                    
            results_all[freq] = results_local
                                
        df          = pd.DataFrame.from_dict(results_all,orient='index')
        fname       = coordinate + '_dependent_L1_L2_bias_high_capacity.csv'
        df.to_csv(_dirs.DIR_RESULT + fname)
#
#
#
    
    if RESIDUALS_ONLY:
    
        results_n = dict()
        results_s = dict()
        for freq in target_freqs:
            print(str(freq))
            freq_index = pydal.utils.find_target_freq_index(freq, f)    
            
            for hydro in p_hydros: 
                hydro = hydro.capitalize()
                if hydro.capitalize() == 'North':
                    rl_true = rl_n 
                if hydro.capitalize() == 'South':
                    rl_true = rl_s 
                rl_true = rl_true[freq_index,:]
                rl_true = np.array(rl_true)
                    
                #Split the data according to how it was trained:
                dset_full               = classes.f_y_orca_dataset(feature, rl_true)
                test_size               = int ( len(feature) * _vars.TEST)
                hold_size               = int ( len(feature) * _vars.HOLDBACK)
                train_size              = len(dset_full) - test_size - hold_size
                _,_,dset_hold    = \
                    torch.utils.data.random_split(
                        dataset     = dset_full,
                        lengths     = [train_size,test_size,hold_size],
                        generator   = fixed_seed  )
                #Now the test vectors at this stage can be recovered:                    
                feature_samp    = feature[dset_hold.indices]
                label_samp      = rl_true[dset_hold.indices]
    
                cutoff          = np.percentile(rl_true,PERCENTILE)
                mask            = label_samp > cutoff
                label_masked    = label_samp[mask]
                feature_masked  = feature_samp[mask]
    
                # 2. Instantiate desired models    
                models_dict = get_models(freq,freq_index,coordinate,standard,hydro,slr_years,nn_1d_layers,nn_1d_nodes)
                
                """
                testing:
                keys = list(models_dict.keys())
                key = keys[3]
                model = models_dict[key]
                """
                for key,model in models_dict.items():
                    # initialize result array (absolute results, not residuals)
                    temp = np.zeros(len(feature_masked)) 
                    temp = torch.tensor(temp)
                    for index in range(len(feature_masked)):
                        t           = feature_masked[index].float()
                        t           = t.reshape((1,1))
                        temp[index] = model.neural_net(t)
                    # if not ('SLR' in key): # scaling is diferernt between slr and nn
                    #     # Only the scaled NN's need this correction.                
                    #     temp = temp * _vars.RL_SCALING
                    temp = temp * _vars.RL_SCALING 
                    temp = temp.detach().cpu().numpy()
        
                    residual = label_masked - temp
        
                    new_key = key \
                        + r'_residual_' \
                        + r'_' + str(freq).zfill(3) 
                    
                    if hydro.capitalize() == 'North':
                        results_n[new_key] = residual
                    if hydro.capitalize() == 'South':
                        results_s[new_key] = residual
                                
        df_s          = pd.DataFrame.from_dict(results_s,orient='index')
        df_n          = pd.DataFrame.from_dict(results_n,orient='index')
        fname_s       = coordinate + '_South_residuals_with_high_capacity.csv'
        fname_n       = coordinate + '_North_residuals_with_high_capacity.csv'
        df_s.to_csv(_dirs.DIR_RESULT + fname_s)
        df_n.to_csv(_dirs.DIR_RESULT + fname_n)

#
#
#

    if TROUBLESHOOT:
    
        freq = 201
        hydro='North'
        standard = 'STANAG'
        slr_years = ['2020']
        nn_1d_layers=[1]
        nn_1d_nodes = [38]
        
        
        
        freq_index = pydal.utils.find_target_freq_index(freq, f)    
        results_local = dict()
        
        hydro = hydro.capitalize()
        if hydro.capitalize() == 'North':
            rl_true = rl_n 
        if hydro.capitalize() == 'South':
            rl_true = rl_s 
        rl_true = rl_true[freq_index,:]
        rl_true = np.array(rl_true)
            
        #Split the data according to how it was trained:
        dset_full               = classes.f_y_orca_dataset(feature, rl_true)
        test_size               = int ( len(feature) * _vars.TEST)
        hold_size               = int ( len(feature) * _vars.HOLDBACK)
        train_size              = len(dset_full) - test_size - hold_size
        _,_,dset_hold    = \
            torch.utils.data.random_split(
                dataset     = dset_full,
                lengths     = [train_size,test_size,hold_size],
                generator   = fixed_seed  )
        #Now the test vectors at this stage can be recovered:                    
        feature_samp    = feature[dset_hold.indices]
        label_samp      = rl_true[dset_hold.indices]

        cutoff          = np.percentile(rl_true,PERCENTILE)
        mask            = label_samp > cutoff
        label_masked    = label_samp[mask]
        feature_masked  = feature_samp[mask]

        # 2. Instantiate desired models    
        models_dict = get_models(freq,freq_index,coordinate,standard,hydro,slr_years,nn_1d_layers,nn_1d_nodes)
        
        """
                xx      = x[dset_hold.indices]
                yy      = y[dset_hold.indices]
        
                plt.scatter(xx,yy)
                plt.scatter(xx[mask],yy[mask],marker='.')
                
                plt.scatter(feature_masked,label_masked)
        """
        keys    = list(models_dict.keys())
        slr     = models_dict[ keys [ 0 ] ]
        nn      = models_dict[ keys [ -1 ] ]

        # SLR
        y_slr = np.zeros_like(feature_masked)
        y_slr = torch.tensor(y_slr)
        for index in range(len(feature_masked)):
            t           = feature_masked[index].float()
            t           = t.reshape((1,1))
            y_slr[index] = slr.neural_net(t)
        # if not ('SLR' in key): # scaling is diferernt between slr and nn
        #     # Only the scaled NN's need this correction.                
        #     temp = temp * _vars.RL_SCALING
        y_slr = y_slr * _vars.RL_SCALING 
        y_slr = y_slr.detach().cpu().numpy()
        
        # NEURAL NET
        y_nn = np.zeros_like(feature_masked)
        y_nn= torch.tensor(y_nn)
        for index in range(len(feature_masked)):
            t           = feature_masked[index].float()
            t           = t.reshape((1,1))
            y_nn[index] = nn.neural_net(t)
        # if not ('SLR' in key): # scaling is diferernt between slr and nn
        #     # Only the scaled NN's need this correction.                
        #     temp = temp * _vars.RL_SCALING
        y_nn = y_nn * _vars.RL_SCALING 
        y_nn = y_nn.detach().cpu().numpy()

        plt.figure()
        plt.scatter(feature_masked,label_masked,marker='.',label='Real Data')
        plt.scatter(feature_masked,y_slr,marker='.',label='SLR')
        plt.scatter(feature_masked,y_nn,marker='.',label='NN')
        plt.legend()
        plt.show()

        del_nn  = y_nn  - label_masked
        del_slr = y_slr - label_masked
        plt.figure()
        plt.scatter(feature_masked,del_slr,marker='.',label='SLR')
        plt.scatter(feature_masked,del_nn,marker='.',label='NN')
        plt.legend()
        plt.show()


        _,nn_MAE                  = pydal.utils.L1_error(
            label_masked, 
            y_nn)
        _,nn_MSE                  = pydal.utils.L2_error(
            label_masked, 
            y_nn)
        _,slr_MAE                  = pydal.utils.L1_error(
            label_masked, 
            y_slr)
        _,slr_MSE                  = pydal.utils.L2_error(
            label_masked, 
            y_slr)

        import pylab
        plt.figure()
        stats.probplot(del_nn, dist="norm", plot=pylab)
        plt.figure()
        stats.probplot(del_slr, dist="norm", plot=pylab)
        
        plt.figure()
        plt.hist(del_slr,density=True,bins=50,label='SLR')
        plt.hist(del_nn,density=True,bins=50,label='NN')
        plt.legend()
        
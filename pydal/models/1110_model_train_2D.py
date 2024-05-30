"""

Starting from test_2d.py in synthetic-data-sets,
build up using real data from trial results

This file is only for 1d model training, 
i.e., use only the y coordinate as feature and 0-mean RLs as labels

"""
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import multiprocessing as mp

# various utils and imports of my own creation
import classes # local to this directory
import functions # local to this directory
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

# For access to the real 0-mean spectral-xy dataset.
import pydal.models.SLR_with_transforms


# Root dir for saving / loading models
ROOT_DIR            = _dirs.DIR_SINGLE_F_2D_NN

# Control flags
TRAIN_MODELS        = False
HYDRO               = 'North'
VISUALIZE_MODELS    = False
FMIN                = 5
FMAX                = 300   



"""
#
#
#
#
#
"""
testing = False # WHAT IT SOUNDS LIKE
"""
#
#
#
#
"""

def train( loader_train, loader_val, batch_size = 100, epochs=5):
    """
    # Same story as train_batch...
    
    loader = train_dataloader
    batch_Size = 100
    epochs = 5
    e = 1
    """        
    # losses = np.zeros(int(epochs*batch_size))
    loss_e_train = []
    loss_e_val = []
    batch_index = 0
    for e in range(epochs):
        losses_t = []
        losses_v = []
        for i,data in enumerate(loader_train):
            x,y = data
            optimizer.zero_grad()

            # # Run forward calculation        
            y_predict = model(x)
            
            # Compute loss.
            loss_t = loss_fn(y_predict, y)
            
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss_t.backward()
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            del x
            del y
            batch_index += 1
            losses_t.append(loss_t)

        for i, data in enumerate(loader_val):
            x,y = data
            # Forward Pass
            outputs = model(x)
            # Find the Loss
            validation_loss = loss_fn(outputs, y)
            # Calculate Loss
            losses_v.append(validation_loss.item())
        
        loss_e_val.append(losses_v)
        loss_e_train.append(losses_t)
        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return loss_e_train,loss_e_val


if __name__ == "__main__":        
    # Repeatability:
    fixed_seed  = torch.manual_seed(_vars.SEED)
    
    fname2019   = r'concatenated_data_2019.pkl'
    fname2020   = r'concatenated_data_2020.pkl'
    data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
    data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
    data        = pydal.utils.concat_dictionaries(data2019,data2020)
    del data2019
    del data2020 # Neither needed anymore.
    
    f           = data['Frequency']
    rl_s        = data['South'] # 2d array, zero mean gram
    rl_n        = data['North'] # 2d array, zero mean gram
    rl_s        = rl_s / _vars.RL_SCALING #normalize to roughly -1/1    
    rl_n        = rl_n / _vars.RL_SCALING #normalize to roughly -1/1    
    x           = data['X'] / _vars.X_SCALING
    y           = data['Y'] / _vars.Y_SCALING
    
    dir_target = functions.set_directory_struct(ROOT_DIR,HYDRO)

    if TRAIN_MODELS:
        
        for HYDRO in _vars.HYDROS:
            if HYDRO.capitalize() == 'North' :
                rl = rl_n
            if HYDRO.capitalize() == 'South' :
                rl = rl_s
                
            for n_layer in _vars.LIST_N_LAYERS_2D:
                for n_node in _vars.LIST_N_NODES_2D:
                    dir_target              = functions.set_directory_struct(ROOT_DIR,HYDRO)
                    dir_target            = dir_target + str(n_layer) + r' layers/'
                    pydal.utils.check_make_dir(dir_target)
                    dir_target            = dir_target + str(n_node) + r' nodes/'
                    pydal.utils.check_make_dir(dir_target)

                    dir_target_losses       = dir_target + r'losses/'
                    pydal.utils.check_make_dir(dir_target)
                    pydal.utils.check_make_dir(dir_target_losses)     
        
                    for freq_targ in f:
                        
                        if freq_targ < FMIN : continue
                        if freq_targ > FMAX : break
                        f_index     = pydal.utils.find_target_freq_index(freq_targ, f)
                
                        # Assign features and labels here. 
                        received_level  = rl[f_index,:] # 1D numpy array
                        labels          = received_level # 1D numpy array
                        
                        # train data
                        dset_full               = classes.f_xy_orca_dataset(x,y, labels)
                        test_size               = int ( len(y) * _vars.TEST)
                        train_size              = len(dset_full) - test_size
                        dset_train,dset_test    = \
                            torch.utils.data.random_split(
                                dataset     = dset_full,
                                lengths     = [train_size,test_size],
                                generator   = fixed_seed  )
                            
                        # Sampler sets the distribution for selection for a batch of data
                        weights = np.ones(len(dset_train)) / len(dset_train)
                        sampler = torch.utils.data.WeightedRandomSampler(
                            weights, 
                            len(dset_train), 
                            replacement=True)
                        dataloader_train = DataLoader(
                                dset_train, 
                                batch_size=_vars.BATCH_SIZE,
                                # sampler = sampler,
                                num_workers = _vars.NUM_ML_WORKER_THREADS) 
                        dataloader_test = DataLoader(
                                dset_test, 
                                batch_size=_vars.BATCH_SIZE,
                                # sampler = sampler,
                                num_workers = _vars.NUM_ML_WORKER_THREADS) 
                        
                        
                        # Instantiate the network (hardcoded in class)
                        model = classes.DeepNetwork_2d(n_layer,n_node)
                        optimizer = optim.Adam(model.parameters(), lr=_vars.LEARNING_RATE)
                        loss_fn = nn.MSELoss()  # mean squared error    
                        
                        losses_t, losses_v = train(
                            dataloader_train,
                            dataloader_test,
                            _vars.BATCH_SIZE,
                            _vars.EPOCHS
                            )

                        d_losses = {'Train' : losses_t, 'Test' : losses_v}
                        pydal.utils.dump_pickle_file(
                            d_losses, 
                            dir_target_losses, 
                            p_fname = str(int(freq_targ)).zfill(4) + '.loss')

                        
                        fname = dir_target +  str(int(freq_targ)).zfill(4) + '.trch'
                        torch.save(model.state_dict(), fname)
                        del model
            
            
    if VISUALIZE_MODELS:   
        """
        Needs to be updated for the 2d data format!
        # 2 layer 14 node
        fname_get = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_2d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/North/2 layers/14 nodes/0052.trch'
        # 3 layer 38 node
        fname_get = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_2d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/North/3 layers/38 nodes/0050.trch'

        fname_get =r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_2d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/North/6 layers/38 nodes/0053.trch'
        """
        # Visualize model result against real data
        #Set up the cartesian geometry
        step = 0.01
        xmin = -0.5 #corresponds to +/- 10m with _vars.X_SCALING = 20m
        xmax = 0.5
        ymin = -1 # Corresponds to +/- 100m with _vars.Y_SCALING = 100
        ymax = 1
        
    
        x_range     = np.arange(xmin,xmax,step=step)
        y_range     = np.arange(ymin,ymax,step=step)
        x_size      = xmax - xmin
        y_size      = ymax - ymin
        x_surface   = np.ones((# just an array of ones to start
            int(y_size/step),
            int(x_size/step)
            )) # dim1 is column index, dim2 is row index
        y_surface   = ( # use array of ones to make the right y surface
            x_surface[:,:].T * np.arange(ymin,ymax,step=step)*-1
            ).T # hackery to use the numpy functions, no big deal
        x_surface   = ( # use array of ones to make the right y surface
            x_surface[:,:] * np.arange(xmin,xmax,step=step)*-1
            ) # hackery to use the numpy functions, no big deal
    
        list_files = os.listdir(dir_target)
        list_files = [x for x in list_files if not x == 'figs']
        for fname in list_files:
            # fname = '0050.trch' # testing only
            # Get the zero-mean data set
            freq_targ   = int(fname.split('.')[0])
            f_index     = pydal.utils.find_target_freq_index(freq_targ, f)
            received_level  = rl[f_index,:] * _vars.RL_SCALING # 1D numpy array
            # GEt the model and create a data set
            fname_get = dir_target + fname
            model       = classes.DeepNetwork_2d(6,38)
            model.load_state_dict(torch.load(fname_get))
            model.eval()
            result      = np.zeros_like(x_surface)

# have model, want to create image (imshow) over 2d coords       

            with torch.no_grad():
                for iy, ix in np.ndindex(x_surface.shape):
                    # Last dimension is iterated over first, i.e. y dimension of array
                    t = y_surface[iy,ix],x_surface[iy,ix]
                    t = torch.tensor(t).float()
                    r = model.neural_net(t) 
                    result[iy,ix] = r

            result      = result * _vars.RL_SCALING
 
            plt.figure();plt.imshow(result,aspect='auto',origin='lower') 
        
            # Plot the model results over the zero mean data
            fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(10,8))
            ax.scatter(y,received_level,color='blue',marker='.',label='real data')
            ax.scatter(y_range,result,color='red',marker='.',label='model fit')
            ax.set_ylim((-25,25))
            ax.legend()
            fig.supxlabel('Y-position in range X-Y system (m)', fontsize=12)
            fig.supylabel('Reconstructed spectral response, dB ref 1 ${\mu}Pa^2 / Hz$', fontsize=12)
            plt.tight_layout()
            plt.savefig(dir_target + r'figs\\' + fname.split('.')[0] + '.png')
            plt.close('all')
    









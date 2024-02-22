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
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

# For access to the real 0-mean spectral-xy dataset.
import pydal.models.SLR_with_transforms


# Control flags
TRAIN_MODELS        = False
HYDRO               = 'North'
VISUALIZE_MODELS    = False
FMIN                = 500
FMAX                = 1000


class f_y_orca_dataset(torch.utils.data.Dataset):
    def __init__(self, x,outputs):
        """
        inputs and outputs are the features and labels as 1D tensors
        """
        self.assign_basis_values (x,outputs)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model inputs and labels are being moved to {self.device} device.\n\n")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        x and y are here inputs and labels, not cartesian
        """
        x = self.inputs[idx]
        y = self.labels[idx]
        x = x.float()
        y = y.float()
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def assign_basis_values(self,x,outputs):
        """
        kwargs are feature identifiers in key-value pairs
        """
        # x = torch.tensor(x) # 
        # self.inputs = torch.column_stack((x))
        x           = torch.tensor(x)
        self.inputs = x.unsqueeze(1)
        self.labels = torch.tensor(outputs)


def convolve1d_unstructured_x_y_data(x,y,kernel_size=_vars.LEN_SMOOTHING):
    """
    Assume x and y are unordered.
    For 1d, would want to arange values by x for x and y.
    
    x is the FEATURE
    y is the LABEL
    """
    x           = np.reshape(x,len(x))
    y           = np.reshape(y,len(y))
    sort_ind    = np.argsort(x)
    # x           = torch.Tensor(x[sort_ind])
    # y           = torch.Tensor(y[sort_ind])
    x           = x[sort_ind]
    y           = y[sort_ind]
    kern        = torch.ones(kernel_size) / kernel_size
    x           = torch.Tensor ( np.convolve(x,kern,mode='valid') )
    y           = torch.Tensor ( np.convolve(y,kern,mode='valid') )
    x,y         = x.unsqueeze(1),y.unsqueeze(1) # set to dimensions [n,1]
    return x,y    

    
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
            # sort by y position and then smooth RL data
            x,y = convolve1d_unstructured_x_y_data(x,y,_vars.LEN_SMOOTHING)
            # x = x.clone()
            # y = y.clone()
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
    # mp.freeze_support()

    # Repeatability:
    fixed_seed  = torch.manual_seed(_vars.SEED)
    
    fname2019   = r'concatenated_data_2019.pkl'
    fname2020   = r'concatenated_data_2020.pkl'
    data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
    data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
    data        = pydal.utils.concat_dictionaries(data2019,data2020)

    f           = data['Frequency']
    rl_s        = data['South'] # 2d array, zero mean gram
    rl_n        = data['North'] # 2d array, zero mean gram
    rl_s        = rl_s / _vars.RL_SCALING #normalize to roughly -1/1    
    rl_n        = rl_n / _vars.RL_SCALING #normalize to roughly -1/1    
    x           = data['X'] / _vars.X_SCALING
    y           = data['Y'] / _vars.Y_SCALING

    if HYDRO == 'North' :
        rl = rl_n
    if HYDRO == 'South' :
        rl = rl_s
    
    # Set up the target directory, create if it doesn't exist.
    # The root directory that differentiates based on _variables.py
    path = _dirs.DIR_SINGLE_F_NN
    if not ( os.path.isdir(path)) : # need to make dir if doesnt exist
        os.mkdir(path)
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    dir_target  = path + dir_spec_subdir # no ending \\ for os.mkdir
    
    dir_target  = path + dir_spec_subdir + r'\\' + HYDRO 
    if not ( os.path.isdir(dir_target)) : # need to make dir if doesnt exist
        os.mkdir(dir_target)
    dir_target = dir_target + r'\\'

    if TRAIN_MODELS:
        
        for freq_targ in f:
            if freq_targ < FMIN : continue
            if freq_targ > FMAX : break
            f_index     = pydal.utils.find_target_freq_index(freq_targ, f)
    
            # Assign features and labels here. 
            received_level  = rl[f_index,:] # 1D numpy array
            labels          = received_level # 1D numpy array
            
            # train data
            dset_full               = f_y_orca_dataset(y, labels)
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
            model = classes.DeepNetwork()
            optimizer = optim.Adam(model.parameters(), lr=_vars.LEARNING_RATE)
            loss_fn = nn.MSELoss()  # mean squared error    
            
            
            losses_t, losses_v = train(
                dataloader_train,
                dataloader_test,
                _vars.BATCH_SIZE,
                _vars.EPOCHS
                )
            
            fname = dir_target +  str(int(freq_targ)).zfill(4) + '.trch'
            torch.save(model.state_dict(), fname)
            del model

    # # Losses.     
    # # plot end of epoch loss profiles:
    # losses_train_unravel    = np.zeros(len(losses_t)) # epoch length
    # index = 0
    # for l in losses_t:
    #     losses_train_unravel[index] = l[-1]           
    #     index += 1
    # losses_val_unravel    = np.zeros(len(losses_v) ) # epoch length
    # index = 0
    # for l in losses_v:
    #     losses_val_unravel[index] = l[-1]            
    #     index += 1
    # plt.figure();
    # plt.plot(losses_train_unravel,label='train loss')
    # plt.plot(losses_val_unravel,label='validation loss')
    # plt.suptitle('End of epoch losses')
    # plt.legend()
    # plt.show()
            
     
    if VISUALIZE_MODELS:       
        # Visualize model result against real data
        #Set up the cartesian geometry
        xmin = -1
        xmax = 1
        ymin = -1
        ymax = 1
    
        x_range     = np.arange(xmin,xmax,step=0.01)
        y_range     = np.arange(ymin,ymax,step=0.01)
        x_size      = xmax - xmin
        y_size      = ymax - ymin
        x_surface   = np.ones((y_size,x_size)) # dim1 is column index, dim2 is row index
        y_surface   = (x_surface[:,:].T * np.arange(ymin,ymax)*-1).T # hackery to use the numpy functions, no big deal
        x_surface   = x_surface[:,:] * np.arange(xmin,xmax)

        list_files = os.listdir(dir_target)
        list_files = [x for x in list_files if not x == 'figs']
        for fname in list_files:
            # Get the zero-mean data set
            freq_targ   = int(fname.split('.')[0])
            f_index     = pydal.utils.find_target_freq_index(freq_targ, f)
            received_level  = rl[f_index,:] * _vars.RL_SCALING # 1D numpy array
            # GEt the model and create a data set
            fname_get = dir_target + fname
            model = classes.DeepNetwork()
            model.load_state_dict(torch.load(fname_get))
            model.eval()
            result      = []
            test        = torch.tensor(y_range)
            with torch.no_grad():
                for t in test:
                    t = t.float()
                    t = t.reshape((1,1))
                    result.append(model.neural_net(t))                
            result      = np.array(result) * _vars.RL_SCALING

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
    
    
    # # # device = "cuda" if torch.cuda.is_available() else "cpu"
    # # # print(f"\n\nTest tensors are being moved to and from {device} device.\n\n")    
        
   

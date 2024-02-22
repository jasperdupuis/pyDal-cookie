"""

Starting from test_2d.py in synthetic-data-sets,
build up using real data from trial results

This file is only for 1d model training, 
i.e., use only the y coordinate as feature and 0-mean RLs as labels

"""
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import multiprocessing as mp

# For access to the real 0-mean spectral-xy dataset.
import pydal.models.SLR_with_transforms

print("USING pytorch VERSION: ", torch.__version__)

# Control randomization
SEED        = 123456

# Define the data split
TRAIN       = 0.9
TEST        = 0.1
HOLDBACK    = 0.0


# Define the hyperparameters
LEARNING_RATE   = 0.05
EPOCHS          = 3
BATCH_SIZE      = 500
NUM_WORKERS     = 2


class DeepNetwork(nn.Module):
    '''
    A simple, general purpose, fully connected network

    The network topology is hardwired here, see __init__
    '''
    def __init__(self,):
        # Perform initialization of the pytorch superclass
        super(DeepNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n\nModel using {self.device} device, tensors to be moved in loops if needed")
        
        neural_net = nn.Sequential(
              nn.Linear(2, 128),
              nn.ReLU(),
              nn.Linear(128, 128),
              nn.ReLU(),
              # nn.Linear(128, 128),
              # nn.ReLU(),
              nn.Linear(128, 1),
              )
        self.neural_net = neural_net.to(self.device)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.neural_net(x) # see nn.Sequential
                
        return x
    

class f_xy_orca_dataset(torch.utils.data.Dataset):
    def __init__(self, x,y,outputs):
        """
        inputs and outputs are the features and labels as 1D tensors
        """
        self.assign_basis_values (x,y,outputs)
        
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

    def assign_basis_values(self,x,y,outputs):
        """
        kwargs are feature identifiers in key-value pairs
        """
        x = torch.tensor(x) # x coordinate is column index!
        y = torch.tensor(y) # y coordinate is row index!
        self.inputs = torch.column_stack((y,x))
        self.labels = torch.tensor(outputs)

    
def train( loader, batch_size = 100, epochs=5):
    """
    # Same story as train_batch...
    
    loader = train_dataloader
    batch_Size = 100
    epochs = 5
    e = 1
    """        
    losses = np.zeros(epochs)
    batch_index = 0
    for e in range(epochs):
        for i,data in enumerate(loader):
            x,y = data
            y = y.unsqueeze(1)
            x = x.clone()
            y = y.clone()
            optimizer.zero_grad()

            # # Run forward calculation        
            y_predict = model(x)
            
            # Compute loss.
            loss = loss_fn(y_predict, y)
            
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            losses[e] = loss
            del x
            del y
            batch_index += 1

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return losses


if __name__ == "__main__":        
    # mp.freeze_support()

    # Repeatability:
    fixed_seed  = torch.manual_seed(SEED)

    freq_targ   = 55

    fname2019   = r'concatenated_data_2019.pkl'
    fname2020   = r'concatenated_data_2020.pkl'
    # data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
    data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)

    data        = data2020 # for now, concat later
    f           = data['Frequency']
    f_index     = pydal.utils.find_target_freq_index(freq_targ, f)
    rl_s        = data['South'] # 2d array, zero mean gram
    rl_n        = data['North'] # 2d array, zero mean gram
    x           = data['X']
    y           = data['Y']
    
    # Assign features and labels here. 
    received_level  = rl_s[f_index,:] # 1D numpy array
    labels          = received_level # 1D numpy array
    
    # train data
    dset_full               = f_xy_orca_dataset(x,y, labels)
    test_size               = int ( len(y) * TEST)
    train_size              = len(dset_full) - test_size
    dset_train,dset_test    = \
        torch.utils.data.random_split(
            dataset     = dset_full,
            lengths     = [train_size,test_size],
            generator   = fixed_seed  )
    train_dataloader = DataLoader(
           dset_train, 
           batch_size=BATCH_SIZE,
           shuffle=True,
           num_workers = NUM_WORKERS) 
    
    # Instantiate the network (hardcoded in class)
    model = DeepNetwork()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()  # mean squared error
    
    losses = train(
        train_dataloader,
        BATCH_SIZE,
        EPOCHS
        )
           
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"\n\nTest tensors are being moved to and from {device} device.\n\n")    
        
    # Visualize test result
    #Set up the cartesian geometry
    xmin = -20
    xmax = 20
    ymin = -100
    ymax = 100
    
    x_range = np.arange(xmin,xmax)
    y_range = np.arange(ymin,ymax)
    x_size = xmax-xmin
    y_size = ymax-ymin
    x_surface = np.ones((y_size,x_size)) # dim1 is column index, dim2 is row index
    y_surface = (x_surface[:,:].T * np.arange(ymin,ymax)*-1).T # hackery to use the numpy functions, no big deal
    x_surface = x_surface[:,:] * np.arange(xmin,xmax)
    
    x_flat = x_surface.flat
    y_flat = y_surface.flat
    results = np.zeros_like(x_flat)
    for index in range(len(x_flat)):
        test = torch.tensor((y_flat[index],x_flat[index]))
        # test = test.cuda()
        res = model.neural_net(test.float())
        results[index] = res
        
    result = np.reshape(results,x_surface.shape)
    
    plt.figure();plt.imshow(result,aspect='auto');plt.colorbar()











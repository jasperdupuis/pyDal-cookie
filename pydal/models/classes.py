# -*- coding: utf-8 -*-
"""

ML classes

"""

import torch
import torch.nn as nn

import pydal._variables as _vars

class SLR_1d():
    def __init__(self,f,m):
        self.f = f
        self.m = m 
        return
    
    def neural_net(self,x):
        """
        Same language as in ANNs for simplicity in other code writing
        
        x is the coordinate over which the SLR was done (y or x)
        
        assumes the input parameter x is already zero-centered.
        """
        return (x * self.m)
        
        

class DeepNetwork_1d(nn.Module):
    '''
    A general purpose, fully connected network

    The network topology is hardwired here, see __init__
    '''
    def __init__(self,n_layer,n_node):
        # Perform initialization of the pytorch superclass
        super(DeepNetwork_1d, self).__init__()
        self.flatten = nn.Flatten()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"\n\nModel using {self.device} device, tensors to be moved in loops if needed")
        
        if n_layer == 1:
            neural_net = nn.Sequential(
                  nn.Linear(1, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, 1),
                  )
        
        
        if n_layer == 2:
            neural_net = nn.Sequential(
                  nn.Linear(1, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, 1),
                  )
        self.neural_net = neural_net.to(self.device)


    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.neural_net(x) # see nn.Sequential
                
        return x
    
    
class DeepNetwork_2d(nn.Module):
    '''
    A general purpose, fully connected network

    The network topology is hardwired here, see __init__
    '''
    def __init__(self,n_layer,n_node):
        # Perform initialization of the pytorch superclass
        super(DeepNetwork_2d, self).__init__()
        self.flatten = nn.Flatten()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n\nModel using {self.device} device, tensors to be moved in loops if needed")
        
        if n_layer == 1:
            neural_net = nn.Sequential(
                  nn.Linear(2, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, 1),
                  )
        
        if n_layer == 2:
            neural_net = nn.Sequential(
                  nn.Linear(2, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, n_node),
                  nn.ReLU(),
                  nn.Linear(n_node, 1),
                  )
        
        if n_layer > 2 :
            modules = []
            modules.append(nn.Linear(2,n_node))
            modules.append(nn.ReLU())
            for n in range(n_layer):
                modules.append(nn.Linear(n_node,n_node))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(n_node,1))
            
            neural_net = nn.Sequential(*modules)
        
            
        self.neural_net = neural_net.to(self.device)


    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.neural_net(x) # see nn.Sequential
                
        return x
   

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

    
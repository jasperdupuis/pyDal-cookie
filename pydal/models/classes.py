# -*- coding: utf-8 -*-
"""

ML classes

"""

import torch
import torch.nn as nn

import pydal._variables as _vars

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
        # print(f"\n\nModel using {self.device} device, tensors to be moved in loops if needed")
        
        neural_net = nn.Sequential(
              nn.Linear(1, _vars.N_HIDDEN_NODES),
               nn.ReLU(),
              nn.Linear(_vars.N_HIDDEN_NODES, _vars.N_HIDDEN_NODES),
              nn.ReLU(),
               nn.Linear(_vars.N_HIDDEN_NODES, _vars.N_HIDDEN_NODES),
               nn.ReLU(),
               # nn.Linear(N_HIDDEN_NODES, N_HIDDEN_NODES),
               # nn.ReLU(),
               #  nn.Linear(N_HIDDEN_NODES, N_HIDDEN_NODES),
               #  nn.ReLU(),
              nn.Linear(_vars.N_HIDDEN_NODES, 1),
              )
        self.neural_net = neural_net.to(self.device)


    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.neural_net(x) # see nn.Sequential
                
        return x
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:52:25 2024

@author: Jasper
"""

import numpy as np

from 1010_model_train_1d import DeepNetwork
from 1010_model_train_1d import convolve1d_unstructured_x_y_data

model = DeepNetwork()
model.load_state_dict(torch.load('C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_single_f/hdf5_spectrogram_bw_1.0_overlap_90/55.0.mdl'))
model.eval()


# Visualize test result
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

model.eval()
test        = torch.tensor(y_range)
result      = []
with torch.no_grad():
    for t in test:
        t = t.float()
        t = t.reshape((1,1))
        result.append(model.neural_net(t))
    

yy, rr      = convolve1d_unstructured_x_y_data(y,rl_s[f_index,:])
plt.figure() ; plt.scatter(yy,rr,marker='.')
plt.plot(y_range,result,color='red')
# # plt.figure() ; plt.scatter(y,np.convolve(received_level,np.ones(11)/11,mode='same'),marker='.')



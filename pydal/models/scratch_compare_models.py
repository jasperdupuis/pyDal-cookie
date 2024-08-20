# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:38:54 2024

@author: Jasper
"""

import torch
import classes

fname_s = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/SOUTH/Y/high capacity/0100.trch'
fname_n = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/hdf5_spectrogram_bw_1.0_overlap_90/NORTH/Y/high capacity/0100.trch'

model_s   = classes.DeepNetwork_1d(2,512)
model_s.load_state_dict(torch.load(fname_s))
model_n   = classes.DeepNetwork_1d(2,512)
model_n.load_state_dict(torch.load(fname_n))



y = torch.tensor(0).float()
y = y.reshape((1,1))

model_s(y)
model_n(y)

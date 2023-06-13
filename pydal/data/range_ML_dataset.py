# -*- coding: utf-8 -*-
"""

A class which allows me to join the range X, Y, Lat, Lon, RL data in to a 
set ready for ML training.


"""

import h5py as h5
import numpy as np
import torch

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs

class Data_Preprocessor():
    """
    A class that houses all the ways I might normalize, regularize, or otherwise
    manipulate data before putting through algorithm.
    """
    
    def normalize_Z(x):
        """
        Put the passed data on the X~Z(0,1) distribution
        .
        """
        s = np.std(x)
        z = (x - np.mean(x) ) / s
        return z


class Range_ML_Dataset(torch.utils.data.Dataset):
    """
    Move data products in to a PyTorch ready dataset class
    which will behave well with ML methods.
    """    
    
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model inputs and labels will be moved to {self.device} device.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        x and y are here inputs and labels, not cartesian
        """
        x = self.inputs[idx,:]
        y = self.labels[idx]
        x = x.float()
        y = y.float()
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y
    
    def load_xy_and_label_data(self,x,y,labels):
        """
        A simple function which assumes x,y,labels are good. i.e.
        already flattened and rationalized/checked for accuracy (no mixed up indices)
        Should be tensor inputs, np.ndarry also OK. nx1 only.
        
        To farm out the generation of the nx1 inputs from a set of iterables,
        (e.g. from a three lists of "real" run data), see "concat_xy_label_data"
        
        NOTE: Hydrophone selection is NOT part of this function!
        """
        self.y = y
        self.x = x
        x = torch.tensor(x)
        y = torch.tensor(y)
        self.inputs = torch.column_stack((y,x))
        self.labels = torch.tensor(labels)
        
    
    def concat_treat_and_load_xy_hydro_data_from_runlist_single_freq(
             self,
             p_freq,
             p_hydro,
             p_normalize,
             p_dir,
             p_run_list):
        """
        This works for a single hydrophone only.
        
        p_dictionary has keys (not used here), where each value of 
        p_dictionary[key] is another dictionary with keys:
            'X'
            'Y'
            'TL_label'
        all of which provide a 1-d array of same length.
        
        Unpack this structure and load using load_xy_and_label_data
        """
        
        
        
        # Find the length of result vectors first - spec_dict unused in this loop.
        length = 0
        for runID in p_run_list:
            # get the spec file. Load it in. Find N for each run.
            spec_dict , N = pydal.utils.get_spectrogram_file_as_dict(
                runID,
                p_dir,
                p_rotation_rad = 0)
            length = length + N        
            
        # Find target freq index, can use any run so take last from above.
        f_index = pydal.utils.find_target_freq_index(p_freq,spec_dict['Frequency'])
           
        # Pre-allocate memory for speed.            
        xx = np.zeros(length,dtype=np.float64)
        yy = np.zeros_like(xx)
        ll = np.zeros_like(xx)        
        
        # Now iterate over loop again, access returned dictionary here.
        # Assign values to the instantiated arrays.
        count = 0 # for indexing results arrays
        failures = dict()
        for runID in p_run_list:
            spec_dict, N = pydal.utils.get_spectrogram_file_single_hydro_f_as_dict(
                f_index,
                p_hydro,
                runID,
                p_dir,
                p_rotation_rad = _vars.TRACK_ROTATION_RADS)
                
            label = p_normalize(spec_dict['Label'])
            if len (label) == N :
                ll[count:count + N] = label
                xx[count:count + N] = spec_dict['X']
                yy[count:count + N] = spec_dict['Y']
                # Apply normalization function
            else: failures[runID] = len(label)
   
        # now prune the ends of the data for failed runs:
        remove_length = 0
        for key,value in failures.items():
            # get the spec file. Load it in. Find N for each run.
            remove_length = remove_length + value
            
        xx = xx [: -1 * remove_length]
        yy = yy [: -1 * remove_length]
        ll = ll [: -1 * remove_length]
        
        self.failures= failures
        self.load_xy_and_label_data(xx, yy, ll)


if __name__ == '__main__':
    hydro = _vars.HYDROPHONE
    
    freq = _vars.FREQS[10]
    freq_str = str(freq).zfill(4)
    
    dirname = _dirs.DIR_SPECTROGRAM
    dirname = dirname + pydal.utils.create_dirname_spec_xy(
        p_fs                = _vars.FS_HYD, 
        p_win_length        = _vars.T_HYD * _vars.FS_HYD,
        p_overlap_fraction  = _vars.OVERLAP)
    dirname = dirname + r'\\'

    import os
    run_list = os.listdir(dirname)
    run_list = [ r.split('_')[0] for r in run_list if 'DRJ' in r.split('_')[0]]
    
    # Define the normalization function to use in the label generation.
    normalize = Data_Preprocessor.normalize_Z
    
    dset = \
        Range_ML_Dataset()

    dset.concat_treat_and_load_xy_hydro_data_from_runlist_single_freq(
        p_freq = freq,
        p_hydro = hydro,
        p_normalize = normalize,
        p_dir = dirname,
        p_run_list = run_list)


    frac_val = _vars.FRACTION_VALIDATION
    frac_test = _vars.FRACTION_TEST
    frac_train = _vars.FRACTION_TRAIN
    
    generator1 = torch.Generator().manual_seed(_vars.SEED)
    dset_val,dset_test,dset_train = torch.utils.data.random_split(dset, [frac_val,frac_test,frac_train], generator=generator1)

    # train data
    train_dataloader = torch.utils.data.DataLoader(
           dset_train, 
           batch_size = _vars.BATCH_SIZE,
           shuffle = True,
           num_workers = _vars.NUM_ML_WORKER_THREADS,
           drop_last = True) # ignores last batch if not exactly the batch length

    # validation data
    validate_dataloader = torch.utils.data.DataLoader(
        dset_val , 
        batch_size = _vars.BATCH_SIZE,
        shuffle = True,
        num_workers = _vars.NUM_ML_WORKER_THREADS,
        drop_last = True) # ignores last batch if not exactly the batch length
    
    
    # Instantiate the network (hardcoded in class)
    # model = DeepNetwork()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\nModel using {device} device, tensors to be moved in loops if needed")
        
    #TODO : MAke this network genration not have magic number and hard coding.
    model = torch.nn.Sequential(
              torch.nn.Linear(2, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 1),
              )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_vars.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()  # mean squared error
    
    epochs = _vars.EPOCHS
    loss_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    batch_index = 0
    
    # TRAINING
    for e in range(epochs):
        for i,data in enumerate(train_dataloader):
            x,y = data
            x = torch.reshape(x,(_vars.BATCH_SIZE,2))
            x = x.to(device)
            y = torch.reshape(y,(_vars.BATCH_SIZE,1))
            y = y.to(device)
            optimizer.zero_grad()

            # Run forward calculation        
            y_predict = model(x)
            
            # Compute loss.
            loss = loss_fn(y_predict, y)
            
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            loss = loss.data.item()
            optimizer.step()
            loss_train[e] = loss

            del x
            del y
            batch_index += 1
            
            
        # VALIDATION
        model.eval()
        temp_loss_list = list()
        for i,data in enumerate(validate_dataloader):
            xv,yv = data
            xv = xv.float()
            xv = torch.reshape(xv,(_vars.BATCH_SIZE,2))
            xv = xv.to(device)
            yv = torch.reshape(yv,(_vars.BATCH_SIZE,1))
            yv = yv.to(device)

            y_pred = model(xv)
            loss = loss_fn(input=y_pred, target=yv)
    
            temp_loss_list.append(loss.detach().cpu().numpy())
        
            del xv
            del yv
        
        loss_val[e] = np.average(temp_loss_list)

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)    
        print("\ttrain loss: %.7f" % loss_train[e])
        print("\tval loss: %.7f" % loss_val[e])
        print ("~~~ ~~~ ~~~")
           
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\nTest tensors are being moved to and from {device} device.\n\n")  
    
    import matplotlib.pyplot as plt
    plt.plot(loss_train, label='Train');plt.plot(loss_val);plt.legend()
    
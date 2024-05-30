# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:54:52 2023

@author: Jasper
"""

import os
import h5py as h5
import numpy as np
import pickle

import pydal.models.SLR_with_transforms as SLR_with_transforms
import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

import math


def rotate(x,y,theta,xo=0,yo=0): 
    #rotate x,y around xo,yo by theta (rad)
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
    return xr,yr


def create_dirname_spec_xy(p_fs,p_win_length,p_overlap_fraction):
    """
    Handle some logic implicit in naming spectrogram data representation,
    and generate a directory name or filename leading string.
    
    WIN_LENGTH IS INTEGER NUMBER OF SAMPLES LENGTH
    """
    t = p_win_length / p_fs
    bw = str ( 1 / t )[:3]
    overlap = str(p_overlap_fraction)
    ovrs = overlap.split('.')

    # Two cases, ovrs has 2 elements or 1 element
    # 2 elements: take the RHS and turn in to percent
    # 1 element: the value should be 00
    if len(ovrs) == 1: 
        overlap = '00'
    if len(ovrs) == 2 :
        overlap = ovrs[1]
        if len(overlap) == 1:  # 0.1, 0.2, etc
            overlap = int(overlap) * 10
            overlap = str(overlap)
        if len(overlap) > 1: # 0.123, 0.999, etc
            overlap = str(overlap[:2])
    dirname = 'hdf5_spectrogram_bw_' \
        + bw \
        + '_overlap_' \
        + overlap

    return dirname


def get_all_runs_in_dir(
        p_dir):
    """
    filenames must begin with runID_blahblahblah to work with this function.
    """
    runs = os.listdir(p_dir)
    runs = [x.split('_')[0] for x in runs]
    runs = [x for x in runs if x[:2] == 'DR']
    return runs


def get_run_selection( 
        p_list_runs,
        p_type = 'DR',
        p_mth = 'J',
        p_machine = 'A',
        p_speed = '05',
        p_head = 'X',
        p_beam = 'B'): #X means both
    """
    Get an unordered list of runs based on simple criteria and return it
    
    Beam can be N, S, or B. B is the most common by far.
    """
    result = []
    
    for runID in p_list_runs:
        typeof      = True
        mth         = False
        mach        = False
        speed       = False
        head        = False
        beam        = False

        # Type selection
        if (p_type not in runID[:2]): 
            continue  # this run doesn't fit selection

        # Month selection (functionally, selection on 2019 or 2020 data)
        if (p_mth == 'X') : # All month selections returned.
            mth = True
        elif (p_mth in runID[2]): 
            mth = True  

        # Machine config selection
        if (p_machine == 'X') : # All machine states.
            mach = True
        elif (p_machine in runID[8]): 
            mach = True
            
        # Nominal speed selection
        if (p_speed == 'X') : # All speeds.
            speed = True
        elif (p_speed in runID[6:8]): 
            speed = True
            
        # Heading selection
        if (p_head == 'X') :  # all headings
            head = True
        elif (p_head in runID[12]): 
            head = True
            
        # Beam vs keel aspect selection
        if p_beam == 'X':
            beam = True
        elif (p_beam in runID[-1]):
            beam = True
            
        if typeof and mth and mach and speed and head and beam:
            result.append(runID)
        
    return result


def get_hdf5_hydrophones_file_as_dict(
        p_run,
        p_dir = _dirs.DIR_HDF5_HYDROPHONE):
    """
    Retrieve stored values, processed from the binary.
    Keys should be North, North label, South, South label
    """
    result = dict()
    fname = p_dir + p_run + r'_range_hydrophone.hdf5'
    h = h5.File(fname)
    for key in list(h.keys()):
        result[key] = h[key][:]
    h.close()
    return result


def _h5_key_value_to_dictionary(dictionary,hdf5):
    """
    Gadget to help read hdf5 files in to a dictionary structure.

    Must be able to recursively handle groups as well as datasets
    
    handle is the hdf5 file reference, opened elsewhere!
    """
    for key in hdf5.keys():
        value = hdf5[key]
        t = type(value)
        if t == h5._hl.group.Group: #Group
            temp = dict()
            result = _h5_key_value_to_dictionary(temp, value)
            dictionary[key] = result
        else: #Dataset
            dictionary[key] = hdf5[key][:][:]
    return dictionary


def get_spectrogram_file_as_dict(
        p_runID,
        p_dir,
        p_rotation_rad = 0):
    """
    For the target runID and root directory create a dictionary with its data
    """
    result          = dict()    
    fname           = p_dir + p_runID + r'_data_timeseries.hdf5'
    h               = h5.File(fname)
    result          = dict()
    result          = _h5_key_value_to_dictionary(result, h)
    h.close()
    
    #need to do some data curation for length of time and spec dimensions.
    if len( result['North_Spectrogram_Time'] ) < len(result['X']): 
        # trim the last xy to make t_n fit.
        result['X'],result['Y'] = result['X'][:-1],result['Y'][:-1]
    
    if len( result['X'] ) < len ( result [ 'North_Spectrogram_Time' ] ): 
        # trim the last xy to make t_n fit.
        result['South_Spectrogram'] = result['South_Spectrogram'] [:,:-1]
        result['North_Spectrogram'] = result['North_Spectrogram'] [:,:-1]
    
    
    if p_rotation_rad == 0 : 
        x = 1 # Do nothing
    else: 
        result['X'],result['Y']     = pydal.utils.rotate(
            result['X'],
            result['Y'], 
            p_rotation_rad)

    N = len(result['X'])
    
    return result, N


def get_spectrogram_file_single_hydro_f_as_dict(
        p_freq_index,
        p_hydro,
        p_runID,
        p_dir,
        p_rotation_rad = 0):
    """
    Reads a spectrogram-XY-time file using pydal.utils.get_spectrogram_file_as_dict
    
    Also adds a new key-value pair based on the passed combination of 
    hydrophone and frequency. The key is "Label"
    """
    spec, N = get_spectrogram_file_as_dict(
        p_runID,
        p_dir,
        p_rotation_rad )
    if p_hydro == 'NORTH':            
        spec['Label'] = spec ['North_Spectrogram'] [ p_freq_index , : ]
    if p_hydro == 'SOUTH':            
        spec['Label'] = spec ['South_Spectrogram'] [ p_freq_index , : ]

    return spec, N
    

def find_target_freq_index(
        p_f_targ,
        p_f_basis):
    """
    Map an int or float real frequency to array index.
    gets the first value where this is true.
    """
    target_index = \
        np.where(p_f_basis - p_f_targ + 1 > 0)[0][0] 
    target_index = target_index 
    return target_index


def load_hdf5_file_to_dict(p_fname):
    """
    A utility that puts each key-value in the hdf5 in to a dictionary object.
    
    p_fname must be fully qualified
    """
    result = dict()
    with h5.File(p_fname, 'r') as file:
        for key in list(file.keys()):
            result[key] = file[key][:]

    return result


def load_target_spectrogram_data(
        p_runID,
        p_data_dir):
   """
   given a runID and data directory load the results to a dictionary.
   
   Remember, the arrays accessed here are in linear values.
   """    
   result = dict()
   fname = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
   with h5.File(fname, 'r') as file:
       try:
           result['North_Spectrogram']      = file['North_Spectrogram'][:]
           result['North_Spectrogram_Time'] = file['North_Spectrogram_Time'][:]
           result['South_Spectrogram']      = file['South_Spectrogram'][:]
           result['South_Spectrogram_Time'] = file['South_Spectrogram_Time'][:]
           result['Frequency']              = file['Frequency'][:]
           if 'AM' not in p_runID :
               result['X'] = file['X'][:]
               result['Y'] = file['Y'][:]
               result['R'] = np.sqrt(result['X']*result['X'] + result['Y']*result['Y']) 
               result['CPA_Index'] = np.where(result['R']==np.min(result['R']))[0][0]
       except:
           print (p_runID + ' didn\'t work')
           print (p_runID + ' will have null entries')
       return result
    
def load_target_spectrogram_data_with_theta_phi(p_runID,p_data_dir):
    """
    For the case where want to load theta as well as xy data.
    """
    result = load_target_spectrogram_data(p_runID,p_data_dir)
    fname = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
    with h5.File(fname, 'r') as file:
        # BELOW NOT FUTURE PROOF, CHANGED 'Theta North' To 'North_Theta' in 
        # script #015 to be  consistent AFTER generation of 
        # data files 2023 09 01.
        # result['North_Theta']            = file['Theta North'][:] * _vars.DEG_TO_RAD
        # result['South_Theta']            = file['Theta South'][:] *_vars.DEG_TO_RAD
        result['North_Theta']            = file['Theta North'][:] * _vars.DEG_TO_RAD
        result['South_Theta']            = file['Theta South'][:] *_vars.DEG_TO_RAD
    return result
    

def insert_hydro_RAM_TL_to_spec_dictionary_from_file(
        p_hydro,
        p_spec_dict,
        p_runID,
        p_data_dir
        ):
    fname   = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
    key     = p_hydro.capitalize() + '_RAM_TL_interpolations'
    subdictionary = dict()
    with h5.File(fname, 'r') as file:
        subdictionary  = _h5_key_value_to_dictionary(subdictionary,file[key])
        p_spec_dict [ key ] = subdictionary
    return p_spec_dict
   
    
def get_gram_XY_index_by_distToCPA(
        p_gram_dict,
        p_distToCPA = 33 ):
    """
    Returns the start and end indices for when the track was within 
    p_distToCPA meters from CPA on either side. 
    """
    x = p_gram_dict['X']
    y = p_gram_dict['Y']
    r_cpa = np.sqrt( x ** 2 + y ** 2 )
    target_xy_index = \
        np.where(r_cpa - p_distToCPA < 0)[0][0] 
    start_index = target_xy_index
    end_index = 2 * start_index
    return start_index, end_index


def get_fully_qual_spec_path():
    """
    Void function as only requires macro definitions from _vars and _dirs
    """
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    p_dir_spec      = _dirs.DIR_SPECTROGRAM + dir_spec_subdir + '\\'
    run_list        = pydal.utils.get_all_runs_in_dir(p_dir_spec)
    return p_dir_spec,run_list


def dump_pickle_file(
        p_dictionary_data,
        p_data_dir,
        p_fname):
    fname = p_data_dir + p_fname
    with open( fname, 'wb' ) as file:
        pickle.dump( p_dictionary_data, file )
        
        
def load_pickle_file(
        p_data_dir,
        p_fname):
    fname = p_data_dir + p_fname
    with open( fname, 'rb' ) as file:
        result = pickle.load( file )
    return result


def get_max_index(xx,yy,ss,nn):
    """
    Do a thorough check of time-base equivalence.
    Provide new index maximum if appropriate
    
    xx,yy are integers representing 1-D array lengths
    ss,nn are tuples (a,b) representing 2d array shape
    
    trimming is done using the result of this function.
    """
    if not ( xx == yy ):
        print('Error in check_sizes, x and y axis different lengths!')
        return 0
    if not ( ss == nn ) :
        print('Error in check_sizes, ss and nn different shapes!')
        return 0
    if ( xx == nn[1] ) and ( xx == ss[1] ) :
        # Everything is good, just return xx (length of x) for indexing.
        return xx    
    if ( xx < nn[1] ) and ( xx < ss[1] ):
        # Need to truncate nn and ss to fit xx
        return xx
    if ( xx > nn[1] ) and ( xx > ss[1] ):
        # Need to truncate xx and yy to fit ss and nn
        return nn[1]
    
def concat_dictionaries(dict1,
                        dict2,
                        excl_list = [ 'Transform', 'Frequency']):
    """
    Assumes dict1 and dict2 have the exact same structure and key strings
    
    Strings in excl_list will not be concatenated
    """
    result = dict()

    for key,value in dict1.items():
        if key in excl_list: continue #ignore specified entries
        result[key]     = np.concatenate((dict1[key],dict2[key]),axis=-1)

    for key in excl_list: # add the skipped stuff, no concat operation
        result[key] = dict1[key]
    return result

def check_make_dir(p_target):
    if os.path.isdir(p_target): return #it already exists
    else:
        os.mkdir(p_target)
        

def load_training_data(p_bool_true_for_dict=False):
    """
    This data has been 
    """
    fname2019   = r'concatenated_data_2019.pkl'
    fname2020   = r'concatenated_data_2020.pkl'
    data2019    = SLR_with_transforms.load_concat_arrays(fname2019)
    data2020    = SLR_with_transforms.load_concat_arrays(fname2020)
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
    
    if p_bool_true_for_dict:
        data['South'] = rl_s        
        data['North'] = rl_n
        data['X'] = x
        data['Y'] = y
        return data
    else:
        return f,rl_s,rl_n,x,y


def L1_error(y,y_hat):
    """
    same length required.
    """
    delta = y_hat-y
    L1 = np.sum(delta) / len(y)
    return L1


def L2_error(y,y_hat):
    """
    
    """
    delta = (y_hat-y)**2
    L2 = np.sum(delta) / len(y)
    return L2
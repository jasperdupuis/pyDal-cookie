# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:54:52 2023

@author: Jasper
"""

import os
import h5py as h5
import numpy as np

import pydal._directories_and_files as _dirs

def create_dirname(p_fs,p_win_length,p_overlap_fraction):
    """
    Handle some logic implicit in naming spectrogram data representation,
    and generate a directory name or filename leading string.
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
    return runs

def get_run_selection( 
        p_list_runs,
        p_type = 'DR',
        p_mth = 'J',
        p_machine = 'A',
        p_speed = '05',
        p_head = 'X'): #X means both
    """
    Get an unordered list of runs based on simple criteria and return it
    """
    result = []
    for runID in p_list_runs:
        if (p_type not in runID[:2]): continue # Type selection
        if (p_mth not in runID[2]): continue # Month (year) selection
        if (p_machine not in runID[8]): continue # Machine config selection
        if (p_speed not in runID[6:8]): continue # peed selection
        if (p_head == 'X') : # Check for heading agnostic first.
            result.append(runID)
            continue
        elif (p_head not in runID[12]): 
            continue # Heading selection
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


def find_target_freq_index(
        p_f_targ,
        p_f_basis):
    """
    Map an int or float real frequency to array index.
    gets the first value where this is true.
    """
    target_index = \
        np.where(p_f_basis - p_f_targ > 0)[0][0] 
    target_index = target_index - 1 #to ensure actually catching entire desired bin.
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

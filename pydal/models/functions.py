# -*- coding: utf-8 -*-
"""

functions

"""

import os

import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars

def set_directory_struct(path,p_hydro):
    # Set up the target directory, create if it doesn't exist.
    # The root directory that differentiates based on _variables.py
    if not ( os.path.isdir(path)) : # need to make dir if doesnt exist
        os.mkdir(path)
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        _vars.FS_HYD,
        _vars.T_HYD_WINDOW * _vars.FS_HYD,
        _vars.OVERLAP
        )
    dir_target  = path + dir_spec_subdir # no ending \\ for os.mkdir
    if not ( os.path.isdir(dir_target)) : # need to make parent if doesnt exist
        os.mkdir(dir_target)
    
    # Now teh hydrophone specification:    
    dir_target  = path + dir_spec_subdir + r'\\' + p_hydro 
    if not ( os.path.isdir(dir_target)) : # need to make dir if doesnt exist
        os.mkdir(dir_target)
    dir_target = dir_target + r'\\'
    
    return dir_target

def set_NN_path(parent,hydro,layer,node):
    path = set_directory_struct(parent,hydro)
    path = path \
        + str(layer) + r' layers\\' \
        + str(node) + r' nodes\\'
    return path


"""

A file to hold all explicit strings for directory and file management

import example:
    
from directories_and_files import DIR_SPETROGRAM, SUMMARY_FNAME

"""

"""

DIRECTORIES

"""
# # Data directory of interest, note there are a few different ways of making spectrograms

# # The generic spectrogram directory:
DIR_SPECTROGRAM = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\spectrograms\\'

# 2019 data only
DIR_BINARY_HYDROPHONE = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'

DIR_HDF5_HYDROPHONE = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\hydrophone_hdf5\\'

# 2019 data only
DIR_TRACK_DATA = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'

DIR_HYDRO_CALS = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\\'

""" 

RELATIVE FILE NAMES 

"""

# Contains kurtosis, tracks, etc.
SUMMARY_FNAME = r'summary_stats_dict.pkl'


"""

EXPLICIT FILE NAMES

"""
# Maps config-speed pairs to lists of runIDs matching those parameters. 
FNAME_CONFIG_TRIAL_MAP =\
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/external/config_speed_run_dictionary.pkl'
# The shaft speed - RPM table in a space-delimited file.
FNAME_SPEED_RPM = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/external/RPM_shaft_map.txt'

TRIAL_MAP = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/burnsi_files_RECONCILE_20201125.csv'

FILE_NORTH_CAL = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging/Range Data Amalg/TF_DYN_NORTH_L_40.CSV'
FILE_SOUTH_CAL = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging/Range Data Amalg/TF_DYN_SOUTH_L_40.CSV'



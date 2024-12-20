"""

A file to hold all explicit strings for directory and file management

import example:
    
from directories_and_files import DIR_SPETROGRAM, SUMMARY_FNAME

"""

"""

DIRECTORIES

"""

YEAR = 2020

# # Data directory of interest, note there are a few different ways of making spectrograms

"""

DATA SOURCE DIRECTORIES

"""
if not (YEAR == 2020):# 2019 data only :
    DIR_RANGE_DSR_RESULT = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN'
    DIR_BINARY_HYDROPHONE = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
    # 2019 data only
    DIR_TRACK_DATA = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'
    DIR_HYDRO_CALS = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\\'
    FILE_NORTH_CAL = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging/Range Data Amalg/TF_DYN_NORTH_L_40.CSV'
    FILE_SOUTH_CAL = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging/Range Data Amalg/TF_DYN_SOUTH_L_40.CSV'
    fname_range_dict = \
        r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/range_info/dynamic_patbay_2019.py'
    RANGE_DICTIONARY = dict()
    with open(fname_range_dict,'r') as f:
        RANGE_DICTIONARY = eval(f.read())

else:# 2020 data :
    DIR_BINARY_HYDROPHONE = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data\ES0453_MOOSE_SPC_DYN\RAW_TIME\\'
    # 2019 data only
    DIR_TRACK_DATA = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data\ES0453_MOOSE_SPC_DYN\TRACKING\\'    
    DIR_HYDRO_CALS = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data\\'
    FILE_NORTH_CAL = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data/TF_DYN_NORTH_L_40.CSV'
    FILE_SOUTH_CAL = \
        r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data/TF_DYN_SOUTH_L_40.CSV'
    fname_range_dict = \
        r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/range_info/dynamic_patbay_2020.py'
    RANGE_DICTIONARY = dict()
    with open(fname_range_dict,'r') as f:
        RANGE_DICTIONARY = eval(f.read())


"""

INTERIM DATA DIRECTORIES (post some processing)

"""
# # The generic spectrogram directory:
DIR_SPECTROGRAM = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\spectrograms\\'

DIR_HDF5_HYDROPHONE = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\hydrophone_hdf5\\'

DIR_HYDRO_CALS = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2019-Orca Ranging\Range Data Amalg\\'

DIR_BELLHOP_DATA = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\BELLHOP_synthetic\\'

DIR_RAM_DATA = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\RAM_synthetic\\'

DIR_SL_RESULTS_LOGR = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\processed\SL results 20logR\\'

DIR_SL_RESULTS_RAM = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\processed\SL results RAM\\'

DIR_THESIS_IMGS = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/results/__Thesis_figures\\'

"""

"""
DIR_RESULT = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\\'
DIR_RESULT_AMBIENTS = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Ambients\\'
DIR_RESULT_SLR =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\SLR\\'
DIR_RESULT_NOISEBAND =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Noiseband\\'
DIR_RESULT_RESIDUALS_AND_BIAS =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Residuals and Bias\\'
DIR_RESULT_LME =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Lloyds Mirror\\'
DIR_RESULT_SSP =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Soundspeed\\'
DIR_RESULT_CORRELATION =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Correlation\\'
DIR_RESULT_PL_MODELS =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\results\\'
DIR_RESULT_PL_MODELS_PATBAY_ONLY =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\UWAEnvTools\results\\'
DIR_RESULT_LOSS_CURVES =\
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\results\Loss Curves\\'
DIR_RESULT_PL_QUANTIFICATION =\
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/results/PL Quantification/'

"""


"""

DIR_DATA_PROP_MODELS = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\PL_model_predictions\\'


"""

MODEL DIRECTORIES

"""

DIR_SINGLE_F_SLR = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\models\saved_models_1d_SLR\\'
DIR_SINGLE_F_1D_NN = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f/'
DIR_SINGLE_F_2D_NN = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_2d_single_f/'
DIR_SINGLE_F_1D_NN_MULTI_EPOCH = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/pydal/models/saved_models_1d_single_f_multi_epoch/'


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



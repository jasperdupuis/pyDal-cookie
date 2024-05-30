"""

Control variables accessible across multiple modules as required.

The intent of this is that I Can now just edit this file to change parameters
instead of managing globals across multiple files

"""

import numpy as np
import os


FMAX = 300 # space issues

"""
LISTS OF USEFUL THINGS
"""
STANDARDS       = ['STANAG', 'ISO']
HYDROS          = ['NORTH','SOUTH']
MTHS            = ['J','F']
YEARS           = ['2019','2020','All']


"""
RUN SELECTION VARIABLES
"""

# An ambient run from July 2019
# TYPE = 'AM'
# MTH = 'J'
# STATE = 'X'
# SPEED='00'        
# HEADING = 'X' #X means both

# A set of dynamic run parameters from July 2019.
# TYPE = 'DR'
# MTH = 'J'
# STATE = 'A'
# SPEED='X'        
# HEADING = 'X' #X means both

# A set of dynamic run parameters from Feb 2020.
TYPE = 'DR'
MTH = 'J'
STATE = 'A'
SPEED='X'        
HEADING = 'X' #X means both


"""
MATH / PHYSICS VARIABLES
"""
DEG_TO_RAD = np.pi / 180 


"""

ML VARIABLES

"""

# CONTROL RANDOMIZATION
SEED        = 123456

# SCALING FOR NORMALIZATION
RL_SCALING          = 50 # some will be outside -1,1 with this number.
X_SCALING           = 20
Y_SCALING           = 100

# PROCEDURAL STUFF, NEEDNT CHANGE OFTEN ( / EVER)
MIN_F_INDEX_ML              = 3 
MAX_F_INDEX_ML              = 998
NUM_ML_WORKER_THREADS       = 1 # higher numbers should but dont work.

# DATA SPLIT
TRAIN       = 0.8
TEST        = 0.1
HOLDBACK    = 0.1


# ML MODEL PARAMETERS
EPOCHS              = 5
BATCH_SIZE          = 2**6
N_HIDDEN_NODES      = 2**9
N_HIDDEN_LAYERS     = 1 # not yet implemented
LEARNING_RATE       = 0.01
N_WORKERS           = 2
LEN_SMOOTHING       = 1


"""

ML MODEL HYPERPARAMS, VARIOUS KINDS

"""
# x- and y-dim only, no frequency 
LIST_N_LAYERS_1D        = [1]
LIST_N_NODES_1D         = [14,20,26,32,38] #,512]
# x and y, or y and f together.
LIST_N_LAYERS_2D        = [2,4,6,8]
LIST_N_NODES_2D         = [14,20,26,32,38] #,512]

"""

ERROR ANALYSIS VARIABLES

"""

Y_LENGTH    = 200
N_Y_BINS    = 20




"""

TRIAL VARIABLES

"""
FS_HYD = 204800
FS_GPS = 10
LABEL_COM = 'COM '
LABEL_FIN = 'FIN '
F_MIN_HYD = 2
F_MAX_HYD = 85000 # reduce datasets a bit, limit is 90k in calibrations.
REF_UPA = 0.000001


"""

DSP PROCSESING VARIABLES ONLY

"""

T_HYD = 1 #window length in seconds, 1Hz bw
# T_HYD = 0.1 #window length in seconds, 10Hz bw
T_HYD_WINDOW = T_HYD
OVERLAP = 0.9
# OVERLAP = 00


"""

LOCATION AND HYDROPHONE SELECTION VARIABLES

"""
TRACK_ROTATION_DEG = 33 # the value by which x y data must be rotated to match lat-lon based xy.
TRACK_ROTATION_RADS = TRACK_ROTATION_DEG * 3.14159 / 180
HYDROPHONE = 'NORTH'
LOCATION = 'Patricia Bay'

RAM_RESULTS_POINT_OR_LINE = 'POINT'



"""

MODEL VARIABLES

"""

RAM_DELTA_R             = 1.
RAM_F_MIN_AVAIL         = 10    # Lowest freq for which I have run Ram TL
RAM_F_MAX_AVAIL         = 599   # Highest freq for which I have run Ram TL
RAM_F_FAILS             = [63]


"""

RUN VARIABLES

"""
FREQS = 10 + np.arange(190)
TARGET_FREQ = 15
NUM_DAY = '1' #all results will filter on trial day number.

DIST_TO_CPA = 33

DAY = TYPE + MTH + NUM_DAY #AMJ3, DRF1, etc.

# These are for 0.1 s windows
# INDEX_FREQ_LOW = 1
# INDEX_FREQ_HIGH = 8999 #90k cutoff

# These are for  1.0s windows
INDEX_FREQ_LOW = 3
INDEX_FREQ_HIGH = 89999 #90k cutoff

INDEX_FREQ_MAX_PROCESSING = 10000 # as a practical matter don't want all data. 
# This should be ~10kHz, rough limit of accelerometers.

"""
Tracking of various kinds of lists of runs:
"""

# Runs that need a closer look before they will batch:
OOPS_DYN = ['DRJ3PB15AX00EN', # There are no track files for these runs.
        'DRJ3PB15AX00WN',
        'DRJ3PB17AX00EN',
        'DRJ3PB17AX00WN',
        'DRF1PB13AA00WB', # doesnt batch
        'DRF1PB17AA00WB', # doesnt batch
        'DRF1PB17AA00EB', # doesnt batch
        'DRF1PB13AA01WB', # something is empty in this run
        'DRF1PB03AA01EB', # doesnt batch
        'DRF2PB13AA02WB', # something is empty in this run
        'DRF2PB05AA00WN', # doesn't batch
        'DRF2PB07AA00WN', # something is empty in this run
        'DRF2PB05AA00ES', # something is empty in this run
        'DRF5PB05AJ00WN', # something is empty in this run
        'DRF5PB05AJ00EB', # doesn't batch
        'DRF1PB03AA00EB', # Fucking milk run, thanks range dudes.
        ] 


OOPS_AMB = [ # runs the range fucked up for sure:
        'AMJ1PB00XX00XX',
        'AMJ1PB00XX01XX',
        'AMJ1PB00XX02XX',
        'AMJ1PB00XX04XX',
        'AMJ1PB00XX05XX',
        'AMJ2PB00XX01XX',
        'AMJ2PB00XX02XX',
        'AMJ3PB00XX00XX']

    
GOOD_AMB = [ # Ambients that appear OK
        'AMJ1PB00XX03XX', 
        'AMJ1PB00XX06XX',
        'AMJ1PB00XX07XX',
        'AMJ1PB00XX08XX',
        'AMJ1PB00XX09XX',
        'AMJ1PB00XX10XX',
        'AMJ1PB00XX11XX',
        'AMJ2PB00XX03XX',
        'AMJ3PB00XX01XX',
        'AMJ3PB00XX02XX']


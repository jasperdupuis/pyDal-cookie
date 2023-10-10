"""

Control variables accessible across multiple modules as required.

The intent of this is that I Can now just edit this file to change parameters
instead of managing globals across multiple files

"""

import numpy as np
import os

"""
MATH / PHYSICS VARIABLES
"""
DEG_TO_RAD = np.pi / 180 



"""

ML VARIABLES

"""

SEED                        = 50 # key for repeatability!
BATCH_SIZE                  = 250
EPOCHS                      = 20
NUM_ML_WORKER_THREADS       = 1 # higher numbers should but dont work.
LEARNING_RATE               = 0.1


FRACTION_VALIDATION         = 0.1
FRACTION_TEST               = 0.2
FRACTION_TRAIN              = 0.7


"""

TRIAL VARIABLES

"""
FS_HYD = 204800
T_HYD = 1 #window length in seconds
T_HYD_WINDOW = T_HYD
FS_GPS = 10
LABEL_COM = 'COM '
LABEL_FIN = 'FIN '
OVERLAP = 0.9

F_MIN_HYD = 2
F_MAX_HYD = 85000 # reduce datasets a bit, limit is 90k in calibrations.
REF_UPA = 0.000001

"""

LOCATION AND HYDROPHONE SELECTION VARIABLES

"""
TRACK_ROTATION_DEG = 33 # the value by which x y data must be rotated to match lat-lon based xy.
TRACK_ROTATION_RADS = TRACK_ROTATION_DEG * 3.14159 / 180
HYDROPHONE = 'NORTH'
LOCATION = 'Patricia Bay'

RAM_RESULTS_POINT_OR_LINE = 'POINT'

fname_range_dict = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/range_info/dynamic_patbay_2019.py'

RANGE_DICTIONARY = dict()
with open(fname_range_dict,'r') as f:
    RANGE_DICTIONARY = eval(f.read())

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
NUM_DAY = '3' #all results will filter on trial day number.

DIST_TO_CPA = 33

# An ambient run from July 2019
# TYPE = 'AM'
# MTH = 'J'
# STATE = 'X'
# SPEED='00'        
# HEADING = 'X' #X means both

# A set of dynamic run parameters from July 2019.
TYPE = 'DR'
MTH = 'J'
STATE = 'A'
SPEED='17'        
HEADING = 'X' #X means both

DAY = TYPE + MTH + NUM_DAY #AMJ3, DRF1, etc.

# These are for 0.1 s windows
# INDEX_FREQ_LOW = 1
# INDEX_FREQ_HIGH = 8999 #90k cutoff

# These are for  1.0s windows
INDEX_FREQ_LOW = 3
INDEX_FREQ_HIGH = 89999 #90k cutoff

"""
Tracking of various kinds of lists of runs:
"""

# Runs that need a closer look before they will batch:
OOPS_DYN = ['DRJ3PB15AX00EN', # There are no track files for these runs.
        'DRJ3PB15AX00WN',
        'DRJ3PB17AX00EN',
        'DRJ3PB17AX00WN',
        # 'DRJ3PB05AX02EB', # These runs generate hdf5 files with 0 size, but don't fail processing somehow.
        # 'DRJ2PB11AX01WB',
        # 'DRJ1PB05BX00WB',
        # 'DRJ1PB19AX00EB',
        # 'DRJ1PB05AX00EB', 'DRJ1PB05AX00WB', 'DRJ1PB07AX00EB', #Fucked these up with track overwrite.
        # 'DRJ1PB07AX00WB', 'DRJ1PB09AX00EB', 'DRJ1PB09AX00WB',
        # 'DRJ1PB11AX00EB', 'DRJ1PB11AX00WB', 'DRJ1PB13AX00EB',
        # 'DRJ1PB13AX00WB', 'DRJ1PB15AX00EB', 'DRJ1PB15AX00WB'
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


import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate

import pydal.utils
import pydal._variables as _vars

import UWAEnvTools
from UWAEnvTools import locations
from UWAEnvTools.environment import Approximations
from UWAEnvTools.singleTL.RAM import read_XY_TL_df,read_interpolate_plot_TL_df

location = 'Patricia Bay'
the_location = locations.Location(location)
dir_ram = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/interim/RAM_synthetic//'
dir_spec = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\interim\spectrograms\hdf5_spectrogram_bw_1.0_overlap_90\\'

f = np.arange(10,250)
freqs = []
for fr in f:
    if fr == 63 : continue
    freqs.append(fr)
freqs = np.array(freqs)

runID = r'DRJ3PB09AX01EB'
runID = r'DRJ2PB15EX00WB'
list_runs = os.listdir(dir_spec)
track_rotation_rads = _vars.TRACK_ROTATION_DEG * np.pi / 180
hyd = 'North' 


# Get the correct bound limits in x and y. This is from Location() so it can go first.
approx = Approximations()
cpa_latlon = (the_location.LAT,the_location.LON)
lat_extent_tuple = the_location.LAT_RANGE_CORRIDOR_TUPLE
lon_extent_tuple = the_location.LON_RANGE_CORRIDOR_TUPLE
x_east = approx.latlon_to_xy(cpa_latlon, (cpa_latlon[0],lon_extent_tuple[1])) # most positive ==> east
x_west = approx.latlon_to_xy(cpa_latlon, (cpa_latlon[0],lon_extent_tuple[0])) #most negative ==> west
y_north = approx.latlon_to_xy(cpa_latlon, (lat_extent_tuple[1],cpa_latlon[1])) # most positive ==> north
y_south = approx.latlon_to_xy(cpa_latlon, (lat_extent_tuple[1],cpa_latlon[1])) # most negative  ==> south
x_lim = x_east[0]
y_lim = y_north[1]


# RECEIVED LEVEL DATA
# Load the target run's spectrogram and x-y data
fname_spec = dir_spec + runID + r'_data_timeseries.hdf5'
h = h5.File(fname_spec)
keys_spec = list(h.keys())
spec_x = h['X'][:][:]
spec_y = h['Y'][:][:]
spec_f = h[keys_spec[0]][:][:]
gram_n = h[keys_spec[3]][:][:]
t_n  = h[keys_spec[4]][:][:]
gram_s = h[keys_spec[7]][:][:]
t_s  = h[keys_spec[8]][:][:]
h.close() # hdf5 File closed.
spec_x,spec_y = pydal.utils.rotate(spec_x, spec_y, track_rotation_rads)
spec_x,spec_y = spec_x[:-1],spec_y[:-1] # trim the last sample to make t_n fit.


# TRANSMISSION LOSS DATA
# NOTE: This is frequency dependent to load!
# RAM model results for a particular frequency.
freq = freqs[-1] # FOR TESTING # TODO : REMOVE this line
freq = 500
spec_f_index = pydal.utils.find_target_freq_index( # find the frequency in the spectrogram array indexing
    freq,
    spec_f)
fname_TL = dir_ram + str(freq).zfill(4) + r'_' + hyd+ '.csv'
X,Y,TL = read_XY_TL_df(fname_TL)

# Visualize this frequency's TL profile over the range extent.
read_interpolate_plot_TL_df(
    fname_TL,
    p_xlim = x_lim,
    p_ylim = y_lim,
    p_nstep = 100,
    p_comex = 100)

TL_interped = \
    interpolate.griddata(
        (X,Y),
        TL,
        (spec_x,spec_y))

RL_n_f_at_0dc = 10 * np.log10 ( gram_n[ spec_f_index , : ] ) \
                - np.mean  (10 * np.log10 ( gram_n [ spec_f_index , : ] ))
TL_n_f_at_0dc = TL_interped - np.mean ( TL_interped )

fig,ax = plt.subplots( figsize = ( 8 , 5 ) )
ax.plot( t_n , RL_n_f_at_0dc , label = 'RL')
ax.plot( t_n , TL_n_f_at_0dc , label = 'TL')
plt.legend()


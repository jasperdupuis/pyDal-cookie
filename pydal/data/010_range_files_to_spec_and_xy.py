# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:08:25 2023

@author: Jasper
"""

import os

import h5py as h5
import scipy.interpolate as interpolate
import scipy.signal as signal
import pandas as pd
import numpy as np
import datetime as dt

import pydal.utils
import pydal._directories_and_files as _dirs
import pydal._variables as _vars


class Range_Track():

    DEG_PER_RAD = 180 / 3.14159

    def __init__(self):
        return

    def load_process_specifications(self,range_dictionary):
        self.DF_LINES_TO_SKIP = int(range_dictionary['Track file lines to skip'])
        self.ROTATION_ANGLE_DEG = float(range_dictionary['Rotation Angle Deg'])
        self.ROTATION_ANGLE_RAD = self.ROTATION_ANGLE_DEG / self.DEG_PER_RAD

    def load_data_track(self,p_full_file_name):
        self.data_track_df = pd.read_csv(p_full_file_name,
                         sep=',',
                         skiprows=self.DF_LINES_TO_SKIP,
                         encoding = "ISO-8859-1")

    def trim_track_data(self,
                        r,
                        prop_x_string='   Prop X   ',
                        prop_y_string='   Prop Y   ',
                        CPA_X=0.57,
                        CPA_Y=-4.98):
        """
        r is the 2D distance from CPA at which COMEX and FINEX are called
        Default RCN operations are 100m. (200m track)
        Sometimes we request 150m or 300m.

        Method: Determine propeller 2D distance to CPA for all recorded timestamps.
        Use the passed r as a filter for +/- value
        COMEX is seconds since midnight for -r
        FINEX is seconds since midnight for +r

        Since this is GPS time (i.e. GMT), can later reconcile
        with accelerometers with 0.1s accuracy

        default CPA values are for Ferguson's cove
        """
        prop_x = self.data_track_df[prop_x_string].values
        prop_y = self.data_track_df[prop_y_string].values
        
        dx = prop_x - CPA_X
        dy = prop_y - CPA_Y
        prop_r = np.sqrt(dx ** 2 + dy ** 2)
        CPA_index = np.argmin(prop_r)

        COM_index = np.argmax(prop_r[:CPA_index] - r < 0)
        FIN_index = np.argmax(prop_r[CPA_index:] - r > 0) + CPA_index
        if FIN_index == CPA_index: #
            FIN_index = len(prop_r) - 1

        self.data_track_df_trimmed = self.data_track_df[COM_index:FIN_index]
        
        str_t = ''
        for s in self.data_track_df_trimmed.columns:
            if ('Time' in s) or ('TIME' in s):
                str_t = s
                break
        self.time_basis_trimmed = self.data_track_df_trimmed[str_t].values
        start_secs_since_midnight = self.time_basis_trimmed[0]
        end_secs_since_midnight = self.time_basis_trimmed[-1]
        
        self.start_time_float = start_secs_since_midnight
        self.start_time = str(dt.timedelta(seconds=start_secs_since_midnight))        
        return start_secs_since_midnight, end_secs_since_midnight - start_secs_since_midnight

    def write_trimmed_to_h5(self,
                        h):
        str_px = ''
        for s in self.data_track_df_trimmed.columns:
            if ('Prop X' in s) or ('PROP X' in s):
                str_px = s
                break
        str_py = ''
        for s in self.data_track_df_trimmed.columns:
            if ('Prop Y' in s) or ('PROP Y' in s):
                str_py = s
                break
        str_t = ''
        for s in self.data_track_df_trimmed.columns:
            if ('Time' in s) or ('TIME' in s):
                str_t = s
                break
        h['track_time_scale_gps_since_midnight'] = self.time_basis_trimmed
        h['track_time_scale_gps_since_midnight'].make_scale('time')
        # x coordinate
        h5_data_string_path = 'track_propeller_x'
        h.create_dataset(h5_data_string_path,
                         data=self.data_track_df_trimmed[str_px] \
                         )
        # y coordinate
        h5_data_string_path = 'track_propeller_y'
        h.create_dataset(h5_data_string_path,
                         data=self.data_track_df_trimmed[str_py] \
                         )
        return


def get_and_interpolate_calibrations(
        p_target_f_limits = (_vars.F_MIN_HYD,_vars.F_MAX_HYD),
        p_fs_hyd = _vars.FS_HYD,
        p_window_t = _vars.T_HYD,
        p_dir_calibrations = _dirs.DIR_HYDRO_CALS,
        p_range_dictionary = _dirs.RANGE_DICTIONARY
        ):
    """
    Interpolate range calibration file from its home frequency basis
    to the one of interest.
    
    The assumption is that variation is slow over analysis 
    bandwidths of interest. Quick plot shows this is true.


    Parameters
    ----------
    p_target_f_basis : TYPE, optional
        DESCRIPTION. The default is np.arange(10,9e4,10).
    p_fs_hyd : TYPE, optional
        DESCRIPTION. The default is _vars.FS_HYD.
    p_window_t : TYPE, optional
        DESCRIPTION. The default is _vars.T_HYD.
    p_dir_calibrations : TYPE, optional
        DESCRIPTION. The default is _dirs.DIR_HYDRO_CALS.
    p_range_dictionary : TYPE, optional
        DESCRIPTION. The default is _vars.RANGE_DICTIONARY.

    Returns
    -------
    scal : TYPE
        DESCRIPTION.
    ncal : TYPE
        DESCRIPTION.

    """
    target_bw = 1 / p_window_t
    target_freq_basis = \
        np.arange(p_target_f_limits[0],p_target_f_limits[1],target_bw )
    fname_n = p_range_dictionary['SENSOR DICTIONARY']['HYDROPHONE 1']['Calibration filename']
    fname_n = p_dir_calibrations + fname_n
    fname_s = p_range_dictionary['SENSOR DICTIONARY']['HYDROPHONE 2']['Calibration filename']
    fname_s = p_dir_calibrations + fname_s

    df_s = pd.read_csv(fname_s,
                       skiprows=p_range_dictionary['AMB CAL South Spectral file lines to skip'],
                       encoding = "ISO-8859-1")
    df_n = pd.read_csv(fname_n,
                       skiprows=p_range_dictionary['AMB CAL North Spectral file lines to skip'],
                       encoding = "ISO-8859-1")
    freqs = df_s[df_s.columns[0]].values
    delta_f_cals = freqs[1] - freqs[0]
    len_conv = int(np.ceil((target_bw / delta_f_cals )))
    s = df_s[df_s.columns[1]].values # Should be AMPL (which is really dB)
    n = df_n[df_n.columns[1]].values # SHould be AMPL (which is really dB)
    # Valid provides results only where signals totally overlap
    sc = np.convolve( s, np.ones(len_conv)/len_conv, mode='valid')
    nc = np.convolve( n, np.ones(len_conv)/len_conv, mode='valid')
    # convoluton chops a bit; append values at the end where change is not interesting.
    delta = np.abs( len( sc ) - len( s ) ) # number of missing samples to add; always -ve so take abs
    last = sc[ -1 ] * np.ones(delta)
    sc = np.append(sc,last)
    nc = np.append(nc,last)
    sfx = interpolate.interp1d( freqs, sc ) #lazy way to do it.
    nfx = interpolate.interp1d( freqs, nc )
    ncal = nfx(target_freq_basis) # these are the results
    scal = sfx(target_freq_basis) # these are the results
    
    return target_freq_basis,scal,ncal


def apply_calibrations_to_spectrogram(p_gram_f,
                                      p_gram,
                                      p_target_freq_basis,
                                      p_cal,
                                      p_ref_value = _vars.REF_UPA):                                      
    min_index = pydal.utils.find_target_freq_index(min(p_target_freq_basis), p_gram_f)
    max_index = pydal.utils.find_target_freq_index(max(p_target_freq_basis), p_gram_f)

    while len ( p_target_freq_basis )  > max_index - min_index:
        max_index += 1
    # Add 1 to above to align p_gram_f and p_target_freq_basis in multiplication below
    # gram = 10*np.log10( p_gram [min_index:max_index,:])    
    # gram = p_cal + gram.T # range fuckup, converts from V^2 / hz to uPa^2 / hz
    # gram = p_ref_value * ( 10 ** ( gram / 10 ) ) 
    
    # do it all in linear domain:
    cal = p_ref_value * ( 10 ** ( p_cal / 10))
    gram = p_gram [min_index:max_index,:]
    gram = p_cal * gram.T
    return gram.T


def align_track_and_hyd_data(
    p_the_run_dictionary,
    label_com = 'COM ',
    label_fin = 'FIN ',
    fs_hyd = _vars.FS_HYD,
    t_hyd = _vars.T_HYD):
    """
    ASSUMPTION: len(hydrophone data time) >= len (gps data time)
    That is, we only need to prune the hydrophone data to match gps data.
    So then in that case there are four cases:
        1) Missing both labels
        2) Missing COM
        3) Missing FIN
        4) Both labels present
    Each case must be treated.
    Further there is the potential that len(hyd_data) / fs > len(gps)/10
    even after truncation!
    In this case find the delta and split it evenly between start and end.    
    
    input dictionary depends on having keys 'North', 'South', 'Time', 'North labels',
    'South labels' (which have the 'COM ' and ' FIN' strings)
    """
    # STEP ONE: Get the labels indices or set them to 0,-1
    try:
        index_com = p_the_run_dictionary['North labels'].index(label_com)
    except:
        index_com = 0
    try:
        index_fin = p_the_run_dictionary['North labels'].index(label_fin)    
    except:
        index_fin = -1
        
    # STEP TWO: Apply the label indices to the hydrophone data.
    if index_fin == -1: # Do not want to multiply -1 by fs.
        start = int(index_com * fs_hyd * t_hyd)
        end = int(index_fin)
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
    else: # index IS meaningful, so use it.
        start = int(index_com * fs_hyd * t_hyd)
        end = int(index_fin * fs_hyd * t_hyd)
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
        
    # STEP THREE: Check if signal lengths are good:
    time_g = p_the_run_dictionary['Time'][-1] - p_the_run_dictionary['Time'][0] # Use this in case samples are missed.
        # Treat time_g for float rounding - only want the first decimal place
    time_g = int(time_g * 10) / 10
    time_h = len(p_the_run_dictionary['North'])/fs_hyd
    if not(time_g == time_h):
        #So, the total hydrophone time is not equal to the total gps time elapsed
        dt = time_h - time_g # +ve ==> hyd time exceeds gps time
        dt = np.round(dt,2)
        trunc_one_ended = int(fs_hyd * dt/2) # amount of data to chop from each end
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ trunc_one_ended : -1 * trunc_one_ended ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ trunc_one_ended : -1 * trunc_one_ended ]
    else:
        # The unlikely case of  gps and hyd times aligning.
        # null operation required
        p_the_run_dictionary['North'] = p_the_run_dictionary['North']        
        p_the_run_dictionary['South'] = p_the_run_dictionary['South']

    return p_the_run_dictionary


def interpolate_x_y(
    p_the_run_dictionary):
    # Now, must make sure there is an x,y sample for each time step.
    # Note there are missing time steps but we know they occured, so 
    # interpolate away!
    # 2x 1d interpolations for each of x, y
    x_function = interpolate.interp1d(
        p_the_run_dictionary['Time'], # x
        p_the_run_dictionary['X'])    # f(x)
    y_function = interpolate.interp1d(
        p_the_run_dictionary['Time'], # x
        p_the_run_dictionary['Y'])    # f(x)
    t = np.arange(
        p_the_run_dictionary['Time'][0], #start
        p_the_run_dictionary['Time'][-1], #stop
        1/_vars.FS_GPS)                   #step
    t = np.round(t,1) #round the array to the 
    p_the_run_dictionary['X'] = x_function(t)
    p_the_run_dictionary['Y'] = y_function(t)
    p_the_run_dictionary['Time'] = t
    return p_the_run_dictionary


def process_h5_timeseries_to_spectrograms_from_run_list(
    p_list_run_IDs,         # Which runs to process
    p_df,                   # The trial dictionary
    p_fs_hyd = _vars.FS_HYD,
    p_window = np.hanning( _vars.FS_HYD * _vars.T_HYD ),
    p_overlap_fraction = _vars.OVERLAP,
    p_range_dictionary = _dirs.RANGE_DICTIONARY,
    p_trial_search = 'DRJ',
    p_hydro_dir = _dirs.DIR_HDF5_HYDROPHONE,
    p_track_dir = _dirs.DIR_TRACK_DATA,
    p_mistakes = _vars.OOPS_DYN,
    p_ref_unit = _vars.REF_UPA):
    """
    Operates on hdf5 files which contains 'North', 'North labels',
    'South' and 'South labels'. Changed 20230317 so binary decoding is done
    in a previous module.

    Can be made to work with any SRJ/DRJ/DRF/SRF etc run ID substring, 
    uses contain so needn't necessarily be the front.
    2019 and 2020 have different dir structures so must be provided.
    
    Stores results as linear arrays!
    """
    dirname = pydal.utils.create_dirname_spec_xy(p_fs_hyd,len(p_window),p_overlap_fraction)
    target_dir = _dirs.DIR_SPECTROGRAM + dirname
    if not ( os.path.isdir(target_dir)) : # need to make dir if doesnt exist
        os.mkdir(target_dir)
    
    target_freq_basis,s_cal,n_cal = get_and_interpolate_calibrations() # defaults are good
    # the calibrations get applied in linear space.
    
    for runID in p_list_run_IDs:
        print(runID)
        if runID in p_mistakes : 
            continue #I  made some mistakes... must reload these trk files properly later
        fname_hdf5 = target_dir + r'\\'+ runID + r'_data_timeseries.hdf5'           
        if os.path.exists(fname_hdf5): 
            hps=1
            # continue # Will usually want to re-write when running a batch.
            os.remove(fname_hdf5)
        temp = dict()
        row = p_df[ p_df ['Run ID'] == runID ]
        
        hydros = pydal.utils.get_hdf5_hydrophones_file_as_dict(runID,p_hydro_dir)
        temp['South'] = hydros['South']        
        temp['South labels'] = hydros['South labels']        
        temp['North'] = hydros['North']
        temp['North labels'] = hydros['North labels']
        
        if runID[:2] == 'DR': # track only matters for dynamic
            fname = p_track_dir + row['Tracking file'].values[0]
            track = Range_Track()
            track.load_process_specifications(p_range_dictionary)
            track.load_data_track(fname)
            if r'DRF' in runID: # 
                """
                need to fix range fuckup
                2019 data is already rotated!
                2020 data is not! 
                fuck me!
                how hard is Bruce's fucking job??
                """
                rotate      = _vars.TRACK_ROTATION_RADS
                x           = track.data_track_df[p_range_dictionary['Propeller X string']]
                y           = track.data_track_df[p_range_dictionary['Propeller Y string']]
                X,  Y       = pydal.utils.rotate(x,y,rotate)
                track.data_track_df[p_range_dictionary['Propeller X string']] \
                    = X
                track.data_track_df[p_range_dictionary['Propeller Y string']] \
                    = Y
                
 
            start_s_since_midnight, total_s = \
                track.trim_track_data(r = p_range_dictionary['Track Length (m)'] / 2,
                    prop_x_string = p_range_dictionary['Propeller X string'],
                    prop_y_string = p_range_dictionary['Propeller Y string'],
                    CPA_X = p_range_dictionary['CPA X (m)'],
                    CPA_Y = p_range_dictionary['CPA Y (m)'])
            df_temp = track.data_track_df_trimmed
                
            temp['X'] = df_temp[ p_range_dictionary['Propeller X string'] ].values
            temp['Y'] = df_temp[ p_range_dictionary['Propeller Y string'] ].values
            temp['Time'] = df_temp[ p_range_dictionary['Time string'] ].values
        
            temp = interpolate_x_y(temp) # make sure the entire time base is represented
            temp = align_track_and_hyd_data(temp) # do some truncation
            
        s1 = np.sum(p_window)
        s2 = np.sum(p_window**2) # completeness - not used by me. STFT applies it.
        #Now the 'grams
        overlap_n = int( p_overlap_fraction * p_fs_hyd )
        f,s_t,s_z = signal.stft(temp['South'],
                              p_fs_hyd,
                              window = p_window,
                              nperseg = len(p_window),
                              noverlap = overlap_n,
                              nfft = None,
                              return_onesided = True)
        f,n_t,n_z = signal.stft(temp['North'],
                              p_fs_hyd,
                              window = p_window,
                              nperseg = len(p_window),
                              noverlap = overlap_n,
                              nfft = None,
                              return_onesided = True)
        s_z = 2 * (np.abs( s_z )**2) / ( s2)        # PSD
        s_z = s_z * s1                              # stft applies 1/s1, reverse this
        n_z = 2 * (np.abs( n_z )**2) / ( s2)        # PSD
        n_z = n_z * (s1)                            # stft applies 1/s1, reverse this
        s_z = apply_calibrations_to_spectrogram(
            f, s_z, target_freq_basis, s_cal, p_ref_unit )
        n_z = apply_calibrations_to_spectrogram(
            f, n_z, target_freq_basis, n_cal, p_ref_unit )
        
        temp['South_Spectrogram'] = s_z
        temp['North_Spectrogram'] = n_z
        temp['South_Spectrogram_Time'] = s_t
        temp['North_Spectrogram_Time'] = n_t
        temp['Frequency'] = target_freq_basis
        
        try:
            os.remove(fname_hdf5)
        except:
            print(runID + ' hdf5 file did not exist before generation')
            
        with h5.File(fname_hdf5, 'w') as file:
            for data_type,data in temp.items():
                # note that not all variable types are supported but string and int are
                file[data_type] = data


if __name__ == '__main__':    

    BATCH_RUN   = True
    MANUAL_RUNS = False
    SINGLE_RUN  = False

    TYPE    = 'DR'
    MTH     = 'F' # J is 2019, F is 2020
    MACHINE = 'X'
    SPEED   = 'X'
    HEAD    = 'X'

    runs_to_do = ['DRJ1PB05AX00EB', 'DRJ1PB05AX00WB', 'DRJ1PB07AX00EB', #Fucked these up with track overwrite.
            'DRJ1PB07AX00WB', 'DRJ1PB09AX00EB', 'DRJ1PB09AX00WB',
            'DRJ1PB11AX00EB', 'DRJ1PB11AX00WB', 'DRJ1PB13AX00EB',
            'DRJ1PB13AX00WB', 'DRJ1PB15AX00EB', 'DRJ1PB15AX00WB']

    if BATCH_RUN:
        local_df = pd.read_csv(_dirs.TRIAL_MAP)
        list_run_IDs = list(local_df[ local_df.columns[1] ].values)
        list_run_IDs_filtered = pydal.utils.get_run_selection(
            list_run_IDs,
            p_type      = TYPE,
            p_mth       = MTH,
            p_machine   = MACHINE,
            p_speed     = SPEED,
            p_head      = HEAD)        
    
        if MANUAL_RUNS:
            list_run_IDs_filtered  = runs_to_do
    
        overlap = _vars.OVERLAP
        window_length = _vars.T_HYD_WINDOW
        
        window = np.hanning(_vars.FS_HYD * window_length)
        fs_hyd = _vars.FS_HYD
        range_dict = _dirs.RANGE_DICTIONARY
        mistakes = _vars.OOPS_DYN
        # mistakes = []
        
        process_h5_timeseries_to_spectrograms_from_run_list(
            list_run_IDs_filtered,
            local_df,
            fs_hyd,
            window,
            overlap,
            range_dict,
            p_hydro_dir = _dirs.DIR_HDF5_HYDROPHONE,
            p_track_dir = _dirs.DIR_TRACK_DATA,
            p_mistakes = mistakes)
    
    if SINGLE_RUN:
        # runID       = r'DRJ2PB03AX01EB'
        runID       = r'DRJ3PB09AX01EB'
        p_hydro_dir         = _dirs.DIR_HDF5_HYDROPHONE
        p_t_hyd             = 1.5
        p_window            = np.hanning(_vars.FS_HYD * p_t_hyd )
        p_overlap_fraction  = 0.9
        p_fs_hyd            = _vars.FS_HYD
        p_ref_unit          = _vars.REF_UPA
    
        target_freq_basis,s_cal,n_cal = get_and_interpolate_calibrations(
            p_window_t = p_t_hyd ) # defaults are good except for testing this variable
    
        temp = dict()
        hydros = pydal.utils.get_hdf5_hydrophones_file_as_dict(runID,p_hydro_dir)
        temp['South'] = hydros['South']        
        temp['South labels'] = hydros['South labels']        
        temp['North'] = hydros['North']
        temp['North labels'] = hydros['North labels']
    
        s1 = np.sum(p_window)
        s2 = np.sum(p_window**2) # completeness - not used by me. STFT applies it.
        #Now the 'grams
        overlap_n = int( p_overlap_fraction * p_fs_hyd )
        """
    
        Using default signal.STFT behaviour and using Heinzel definitions of scale
        factors, the returned complex array s_z or n_z is equal to X[k] / s1
        
        That is, it is a magnitude spectrum.
            
        return_onsided = True SHOULD mean needn't include factor of 2 in my processing.
        However inspection of scipy.signal._spectral_helper and other subfunctions
        shows that multiplying X[k] by two is still necessary. BUT don't want
        to include s1 in that factor of two.
        
        To convert returned X[k] / s1 to power spectrum, must multiply by 2 after
        taking out the factor s1, THEN square s_z or n_z.
        
        To get power spectral density from power spectrum, 
        scale as follows :
            
        PSD  = PS / ENBW = PS / ( ( fs * S2 ) / (s1**2) )
        
        """
        # f,s_t,s_z = signal.stft(temp['South'], # only need north for testing.
        #                       p_fs_hyd,
        #                       window = p_window,
        #                       nperseg = len(p_window),
        #                       noverlap = overlap_n,
        #                       nfft = None,
        #                       return_onesided = True,
        #                       scaling = 'spectrum')
        f,n_t,n_z = signal.stft(temp['North'],
                              p_fs_hyd,
                              window = p_window,
                              nperseg = len(p_window),
                              noverlap = overlap_n,
                              nfft = None,
                              return_onesided = True,
                              scaling = 'spectrum')
        ps_to_psd_divisor   = ( p_fs_hyd * s2 ) / (s1 ** 2)
        # s_z = 2 * (np.abs( s_z )**2) / ( s2)              # PSD
        # s_z = s_z * s1                                    # 
        n_z =  ( n_z * s1 )                                 # X[K] (stft applies X[k]/s1, reverse this)
        n_ps = 2 * ( np.abs( n_z ) ** 2 ) / ( s1 ** 2 )     # POWER SPECTRUM (PS)
        n_psd = n_ps / ps_to_psd_divisor                    # POWER SPETRAL DENSITY (PSD)
        n_z = n_psd
        
        # s_z = apply_calibrations_to_spectrogram(
        #     f, s_z, target_freq_basis, s_cal, p_ref_unit )
        n_z = apply_calibrations_to_spectrogram(
            f, n_z, target_freq_basis, n_cal, p_ref_unit )
        n_res = np.mean(n_z[:,30:60],axis=1)
        
        import matplotlib.pyplot as plt #visualization import only
        fname = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/2019-Orca Ranging/Range Data Amalg/ES0451_MOOSE_OTH_DYN/RUN_ES0451_DYN_041_000_EAST_Nhyd_PORT_NB.CSV'
        n_TL = 41.109 # hard read from range file
        df = pd.read_csv(fname,skiprows=61)
        range_f = df[df.columns[0]].values[:len(n_res)]
        range_x = df[df.columns[1]].values[:len(n_res)]
    
        my_f = f[:len(n_res)]
        my_x = 10 * np.log10 ( n_res / _vars.REF_UPA ) + n_TL
        plt.figure()
        plt.plot(my_f,my_x,label='My result');plt.xscale('log')
        plt.plot(range_f,range_x,label='Range result');plt.xscale('log')
        plt.legend()
    
    
        delta = np.mean(range_x - my_x)
        plt.f
        plt.plot(range_f,range_x - my_x);plt.xscale('log')

    


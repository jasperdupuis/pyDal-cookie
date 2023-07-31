# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:38:51 2023

@author: Jasper
"""

import os
import calendar
import sys
import struct
import pandas as pd
import numpy as np
import h5py as h5
import datetime as dt


import pydal._directories_and_files as _dirs
import pydal._variables as _vars


# # Legacy classes, from pydrdc.
# # Do not bother with them, can ignore. For binary conversion and 
# # track alignment only.
class Range_Sensor():
    '''
    Base definition.
    Derived, nation- and range- specific implementations must deal with these attributes and methods.
    '''
    timeseries_floats_uncal = 'not yet set'
    spectrum_df_calibration_unmodified = 'not yet set'  # as range provides
    spectrum_df_calibration_interpreted = 'not yet set'  # TODO: as derived from above for new sample window
    spectrum_df_NB = 'not yet set'
    spectrum_df_OTO = 'not yet set'

    comex_dt = 'not yet set'
    finex_dt = 'not yet set'
    
    def __init__(self):
        return

    def load_range_specifications(self,range_info_dictionary):
        pass

    def load_all_data(self,
                      p_filepaths_dictionary,
                      hp_info_dictionary
                      ):
        """
        :param p_filename_dictionary: key,full_file_paths
        'raw file'
        'calibration file'
        'NB file'
        'OTO file '
        :return:
        """
        hyd_label = hp_info_dictionary['Short Description']
        if (hyd_label+'_Hydrophone_raw' in p_filepaths_dictionary.keys()):
            self.load_data_raw_single_hydrophone(p_filepaths_dictionary[hyd_label+'_Hydrophone_raw'])
        if (hyd_label+'_Hydrophone_Cal') in p_filepaths_dictionary.keys():
            self.load_calibration_single_hydrophone(p_filepaths_dictionary[hyd_label+'_Hydrophone_Cal'])
        if (hyd_label+'_Hydrophone_NB') in p_filepaths_dictionary.keys():
            self.load_data_NB_single_hydrophone(p_filepaths_dictionary[hyd_label+'_Hydrophone_NB'])
        if (hyd_label+'_Hydrophone_OTO') in p_filepaths_dictionary.keys():
            self.load_data_TO_single_hydrophone(p_filepaths_dictionary[hyd_label+'_Hydrophone_OTO'])

    def load_data_raw_single_hydrophone(self,p_full_file_name):
        pass

    def load_calibration_single_hydrophone(self,p_full_file_name):
        pass

    def load_data_NB_single_hydrophone(self,p_full_file_name):
        pass

    def load_data_TO_single_hydrophone(self,p_full_file_name):
        pass

    def write_to_h5(self,
                    h,      #the hdf5 handle
                    key):   #the string handle, range_dictionary.item()['Description']
        pass

    def read_from_h5(self,
                     h,
                     key):
        pass


class Range_Hydrophone_Canada(Range_Sensor):

    BYTES_PER_INT32 = 4

    def __init__(self,
                 p_fs = 204800,
                 p_window_size_time_time = 1.5):
        # Time lookup dictionaries
        x = list(calendar.month_abbr)
        y = list(np.arange(0, 13))
        self.MTH_STRING_TO_INT_DICT_HYDROPHONE = dict(zip(x, y))

    @classmethod
    def from_dict(cls,p_dictionary):
        fs = int(p_dictionary['Sample requency'])
        window_size_time = float(p_dictionary['Window Length (s)'])
        return cls(fs,
                   window_size_time)

    def write_to_h5(self,
                    h,      #the hdf5 handle
                    key):
        """
        Overrides base method.
        """
        if not ('hyd_NB_f_scale' in h):  # Make the axis information.
            h['hyd_NB_f_scale'] = self.spectrum_df_NB.to_numpy()[:, 0]
            h['hyd_NB_f_scale'].make_scale('frequency')
            h['hyd_NB_time_scale'] = np.array([0])
            h['hyd_NB_time_scale'].make_scale('time')

            h['hyd_OTO_f_scale'] = self.spectrum_df_OTO.to_numpy()[:, 0]
            h['hyd_OTO_f_scale'].make_scale('frequency')
            h['hyd_OTO_time_scale'] = np.array([0])
            h['hyd_OTO_time_scale'].make_scale('time')

            h['hyd_calibration_f_scale'] = self.spectrum_df_calibration_unmodified.to_numpy()[:, 0]
            h['hyd_calibration_f_scale'].make_scale('frequency')
            h['hyd_calibration_time_scale'] = np.array([0])
            h['hyd_calibration_time_scale'].make_scale('time')

            h['hyd_raw_time_scale'] = np.arange(
                0,
                len(self.timeseries_floats_uncal_trimmed) \
                ) \
                                      / self.fs
            h['hyd_raw_time_scale'].make_scale('time')

        # The underlying processing returns mag^2 / Hz, but CDF requires 20 log (mag / Hz^.5) = 10 log ( mag^2 / hz)
        # The range already reports this in dB, don't change it.
        h5_data_string_path = key.replace(' ', '_') + r'_NB_results'
        # h.create_dataset(h5_data_string_path, data= 10*np.log10(sensor.spectrum_df_NB.to_numpy()))
        h.create_dataset(h5_data_string_path, data=(self.spectrum_df_NB.to_numpy()[:, 1] \
                                                    .reshape(len(self.spectrum_df_NB.to_numpy()[:, 1]), 1)) \
                         .transpose()
                         )
        h[h5_data_string_path].dims[0].label = 'time'
        h[h5_data_string_path].dims[0].attach_scale(h['hyd_NB_time_scale'])
        h[h5_data_string_path].dims[1].label = 'frequency'
        h[h5_data_string_path].dims[1].attach_scale(h['hyd_NB_f_scale'])

        h5_data_string_path = key.replace(' ', '_') + r'_OTO_results'
        h.create_dataset(h5_data_string_path, data=(self.spectrum_df_OTO.to_numpy()[:, 1] \
                                                    .reshape(len(self.spectrum_df_OTO.to_numpy()[:, 1]), 1)) \
                         .transpose()
                         )
        h[h5_data_string_path].dims[0].label = 'time'
        h[h5_data_string_path].dims[0].attach_scale(h['hyd_OTO_time_scale'])
        h[h5_data_string_path].dims[1].label = 'frequency'
        h[h5_data_string_path].dims[1].attach_scale(h['hyd_OTO_f_scale'])

        h5_data_string_path = key.replace(' ', '_') + r'_uncalibrated_time_floats'
        h.create_dataset(h5_data_string_path, data=self.timeseries_floats_uncal_trimmed)
        h[h5_data_string_path].dims[0].label = 'time'
        h[h5_data_string_path].dims[0].attach_scale(h['hyd_raw_time_scale'])

        h5_data_string_path = key.replace(' ', '_') + r'_calibration_spectrum'
        h.create_dataset(h5_data_string_path, data=(self.spectrum_df_calibration_unmodified.to_numpy()[:, 1] \
                                                    .reshape(
            len(self.spectrum_df_calibration_unmodified.to_numpy()[:, 1]), 1)) \
                         .transpose()
                         )
        h[h5_data_string_path].dims[0].label = 'time'
        h[h5_data_string_path].dims[0].attach_scale(h['hyd_calibration_time_scale'])
        h[h5_data_string_path].dims[1].label = 'frequency'
        h[h5_data_string_path].dims[1].attach_scale(h['hyd_calibration_f_scale'])

    def trim_to_COM_and_FIN(self,
                            gps_time_elapsed_seconds = False):
        """
        Three general COM/FIN cases:
            
            1) Neither COM nor FIN marker in the binary.
            2) Only one of COM or FIN markers in the binary.
            3) Both COM and FIN markers in the binary.
        
        Also two other behaviours to intersect,
            alpha) gps_time_elapsed_seconds = False
        or
            bravo) gps_time_elapsed_seconds = a number.
            
    
        For 3, naively assume COM and FIN markers in the binary file 
        are sufficient to process the run. (assumes they are accurate)
        
        For 1 and 2, these will only occur if at least one exception is thrown:
            
            For 2, if EITHER exception throws, use the good binary marker, the
            passed gps_time_elapsed_seconds, and figure out the other marker index.
            
            For 1, if BOTH exceptions throw, assume COMEX_label_index = 0
            and then use gps_time_elapsed_seconds to find a suitable FINEX,
            picking FINEX_label_index = len(self.trial_time_labels) -1
            if it exceeds the length.
        
        gps_time_elapsed_seconds to n windows is rounded DOWN in modulo division.
        """
        COM_found = True
        FIN_found = True
        
        try:
            COMEX_label_index = self.trial_time_labels.index('COM ')
        except:
            COM_found = False
            print("COM label missing from range data for this run and hydrophone combination.")
            COMEX_label_index = 0
        try:
            FINEX_label_index = self.trial_time_labels.index('FIN ')
        except:
            FIN_found = False
            print("FIN label missing from range data for this run and hydrophone combination.")
            FINEX_label_index = len(self.trial_time_labels) -1
        
        #Handle the cases identified above, in reverse order.
        # The naive processes, case 3 and case alpha.
        if ((COM_found and FIN_found)               #Case 3
            or (gps_time_elapsed_seconds==False)):  #Case alpha
            COMEX_sample_index = int(COMEX_label_index * self.fs * self.window_size_time)
            FINEX_sample_index = int(FINEX_label_index * self.fs * self.window_size_time)            
        
        # For the remaining cases (which are all case bravo), need to reduce
        # gps_time_elapsed_seconds to the nearest 1.5 second interval.
        num_windows = gps_time_elapsed_seconds//self.window_size_time
        
        #case 2a: COM is good
        if ((COM_found) and not(FIN_found)):
            FINEX_label_index = int(COMEX_label_index + (num_windows - 1))
            COMEX_sample_index = int(COMEX_label_index * self.fs * self.window_size_time)
            FINEX_sample_index = int(FINEX_label_index * self.fs * self.window_size_time)
                       
        #case 2b: FIN is good
        if (not(COM_found) and (FIN_found)):
            COMEX_label_index = int(FINEX_label_index - (num_windows - 1))
            COMEX_sample_index = int(COMEX_label_index * self.fs * self.window_size_time)
            FINEX_sample_index = int(FINEX_label_index * self.fs * self.window_size_time)
            
        #case 1: Neither COM nor FIN is good
        if (not(COM_found) and not(FIN_found)):
            COMEX_label_index = 0
            FINEX_label_index = int(COMEX_label_index + (num_windows - 1))
            COMEX_sample_index = 0
            FINEX_sample_index = int(FINEX_label_index * self.fs * self.window_size_time)
            if FINEX_sample_index > len(self.timeseries_floats_uncal):
                FINEX_sample_index = \
                    len(self.timeseries_floats_uncal) - int((self.fs * self.window_size_time)) 
                    # subtraction here allows same handling as other cases

        FINEX_sample_index = FINEX_sample_index +int(self.fs * self.window_size_time)
        self.timeseries_floats_uncal_trimmed = self.timeseries_floats_uncal[COMEX_sample_index:FINEX_sample_index]
        return
        

    def load_range_specifications(self,range_info_dictionary):
        self.fs = int(range_info_dictionary['Hydrophone Sample Frequency'])
        self.window_size_time = float(range_info_dictionary['Window Length (s)'])

        self.FILE_HEADER_BYTE_COUNT = int(range_info_dictionary['File Header Byte Count'])
        self.CHUNK_HEADER_BYTE_COUNT = int(range_info_dictionary['Chunk Header Byte Count'])

        self.INTEGER_SAMPLES_PER_CHUNK = int\
                (
            self.fs \
            * self.window_size_time
                )
        self.BYTE_SAMPLES_PER_CHUNK = int\
            (# provided in bytes representing i32
                self.INTEGER_SAMPLES_PER_CHUNK \
                * self.BYTES_PER_INT32
            )
        self.DATACHUNK_SIZE = int\
            (
                self.BYTE_SAMPLES_PER_CHUNK \
                + self.CHUNK_HEADER_BYTE_COUNT
            )
        self.nb_line_skip = range_info_dictionary['NB Spectral file lines to skip']
        self.oto_line_skip = range_info_dictionary['OTO Spectral file lines to skip']
        # Have to manage the range's fucked up cases here:
        # TODO: This should be based on hydrophone node, not range node.
        if range_info_dictionary['Name'] == 'Pay Bat':
            self.cal_line_skip_1 = range_info_dictionary['AMB CAL South Spectral file lines to skip']
            self.cal_line_skip_2  = range_info_dictionary['AMB CAL North Spectral file lines to skip']
        else:
            self.cal_line_skip_1 = range_info_dictionary['CAL Spectral file lines to skip']
            self.cal_line_skip_2 = self.cal_line_skip_1
            
    def load_data_raw_single_hydrophone(self,p_full_file_name):
        allContentsBytes, chunkContentsBytes, dataLength, numberDataChunks = \
            self._read_range_binary_file(p_full_file_name)
        # HYDROPHONE_GAIN, GainCheckValue = ReadDCGainBytes(allContentsBytes[0:2])
        # maxVolt_test = ReadMaxVoltageBytes(contents[2:10])

        uncalibratedDataFloats = np.zeros(
            numberDataChunks * self.INTEGER_SAMPLES_PER_CHUNK)  # entire run will go here
        labelFinder = []  # identifies windows where COM, CPA, FIN are.

        message = 'All chunks were INTEGER_SAMPLES_PER_CHUNK long'
        for chunkID in range(numberDataChunks):
            data = chunkContentsBytes[
                   int(chunkID * self.DATACHUNK_SIZE): int((chunkID + 1) * self.DATACHUNK_SIZE)]
            if not(len(data) == (self.BYTE_SAMPLES_PER_CHUNK + self.CHUNK_HEADER_BYTE_COUNT)):
                message = 'Chunk number '+ str(chunkID) + ' was not INTEGER_SAMPLES_PER_CHUNK long'
                break
            unCalData, status = self._decode_data_chunk(data)
            uncalibratedDataFloats[int(self.INTEGER_SAMPLES_PER_CHUNK * chunkID):int(
                self.INTEGER_SAMPLES_PER_CHUNK * (chunkID + 1))] = unCalData
            labelFinder.append(status)

        self.timeseries_floats_uncal = uncalibratedDataFloats
        self.trial_time_labels = labelFinder

        return uncalibratedDataFloats, labelFinder, message


    def load_calibration_single_hydrophone(self,
                                           p_full_file_name,
                                           p_hyd = 'none'):
        if self.cal_line_skip_2 == self.cal_line_skip_1:
            df = self._read_range_hydrophone_processed_data(p_full_file_name,self.cal_line_skip_1)
        elif p_hyd == 'one' : df = self._read_range_hydrophone_processed_data(p_full_file_name,self.cal_line_skip_1)
        elif p_hyd == 'two' : df = self._read_range_hydrophone_processed_data(p_full_file_name,self.cal_line_skip_2)
        else: raise ValueError("pydrdc Error in load_calibration_single_hydrophone \n Need to specify which hyd (p_hyd = 'one' or 'two') as they have different line skips for calibration files. Two hydrophone range assumed.")
        values = df[df.columns[1]]
        freqs = df[df.columns[0]]
        #comex, finex = self.read_range_hydrophone_processed_timestamp(p_string_manager,'TO')

        result = pd.DataFrame({ 'Frequency':freqs,
                                'Value':values})
        self.spectrum_df_calibration_unmodified = result
        return freqs, values#, comex, finex


    def load_data_NB_single_hydrophone(self,p_full_file_name):
        df = self._read_range_hydrophone_processed_data(p_full_file_name,self.nb_line_skip)
        values = df[df.columns[1]]
        freqs = df[df.columns[0]]
        #comex, finex = self.read_range_hydrophone_processed_timestamp(p_string_manager,'TO')

        result = pd.DataFrame({'Frequency': freqs,
                               'Value': values})
        self.spectrum_df_NB = result

        return freqs, values#, comex, finex


    def load_data_TO_single_hydrophone(self,p_full_file_name):
        df = self._read_range_hydrophone_processed_data(p_full_file_name,self.oto_line_skip)
        values = df[df.columns[1]]
        freqs = df[df.columns[0]]
        #comex, finex = self.read_range_hydrophone_processed_timestamp(p_string_manager,'TO')

        result = pd.DataFrame({'Frequency': freqs,
                               'Value': values})
        self.spectrum_df_OTO = result

        return freqs, values#, comex, finex


    def _read_range_hydrophone_processed_data(self, p_full_file_name, p_line_skip):
        '''
        p_type = CAl, NB, or OTO accepted arguments.
        Just pd.read with a skipline argument.
        Given a target range csv, get its corresponding data and return it.
        For TF and spectra:    p_linesToSkip = 73
        For track files:       p_linesToSkip = 32
        '''
        df = pd.read_csv(p_full_file_name,
                         sep=',',
                         skiprows=p_line_skip)
        return df


    def read_range_hydrophone_processed_timestamp(self,p_full_file_name):
        '''
        return comex_dt, finex_dt
    
        This is read from the OTO or NB result file; not necessarily useful.
        '''
        
        with open(p_full_file_name) as reader:
            comex = reader.readline()
            finex = reader.readline()
        # Find the comex datetime
        ss = comex.split(' ')
        c_yyyy = int(ss[5][:4])
        c_MM = self.MTH_STRING_TO_INT_DICT_HYDROPHONE[ss[3]]
        c_dd = int(ss[4].split(',')[0])
        c_hh = int(ss[5][5:7])
        c_mm = int(ss[5][8:10])
        c_ss = int(ss[5][11:13])
        c_z = int(ss[5][14])
        comex_dt = dt.datetime(year=c_yyyy,
                               month=c_MM,
                               day=c_dd,
                               hour=c_hh,
                               minute=c_mm,
                               second=c_ss,
                               microsecond=int(c_z * 1e5))
        # Find the finex datetime
        ss = finex.split(' ')
        f_yyyy = int(ss[5][:4])
        f_MM = self.MTH_STRING_TO_INT_DICT_HYDROPHONE[ss[3]]
        f_dd = int(ss[4].split(',')[0])
        f_hh = int(ss[5][5:7])
        f_mm = int(ss[5][8:10])
        f_ss = int(ss[5][11:13])
        f_z = int(ss[5][14])
        finex_dt = dt.datetime(year=f_yyyy,
                               month=f_MM,
                               day=f_dd,
                               hour=f_hh,
                               minute=f_mm,
                               second=f_ss,
                               microsecond=int(f_z * 1e5))
        dictionary = dict()
        dictionary['Header COMEX'] = comex

        self.comex_dt = comex_dt
        self.finex_dt = finex_dt

        return comex_dt, finex_dt

    def _read_range_binary_file(self,p_full_file_name):
        with open(p_full_file_name, mode='rb') as file:
            contents = file.read()
            chunkContents = contents[self.FILE_HEADER_BYTE_COUNT:]
            length_contents = int(len(contents))
            numChunks = int((length_contents - self.FILE_HEADER_BYTE_COUNT) / self.DATACHUNK_SIZE)
        return contents, chunkContents, length_contents, numChunks

    def _decode_data_chunk(self, p_dataBytes):
        """
        status = decodeStatus[0:4]
        sequenceNumber = decodeSequenceNumber[4:8]
        voltageRange = decodeCardRange[8:16]
        data = decodeDataChunk[16:]
        """
        try:
            status = p_dataBytes[0:4].decode("utf-8")  
        except:
            x = 'Error in _decode_data_chunk, error processing status bytes to utf-8.'
        sequenceNumber = p_dataBytes[4:8]
        voltageRange = p_dataBytes[8:16]
        byteData = p_dataBytes[16:]
        data = np.zeros([self.INTEGER_SAMPLES_PER_CHUNK],dtype='int')
        for i in range(self.INTEGER_SAMPLES_PER_CHUNK):
            try:
                data[i] = struct.unpack('>i', byteData[int(i * 4):int((i + 1) * 4)])[0]
            except:
                x = 1
        data_float = data.astype('float64') / 20e6
        return data_float, status

    def _read_DC_gain_bytes(p_FirstTwoBytes):
        """
        0 ==> 20dB
        non-0 ==> 40dB
        """
        value = struct.unpack('h', p_FirstTwoBytes)
        if value == 0: result = 40
        if (not (value == 0)): result = 20
        return result, value

    def _read_V_max_bytes(p_8Bytes):
        """
        Flattened double-float string. Whatever that means.
        """
        result = p_8Bytes.decode(sys.stdout.encoding)
        return result


    def _decode_status_bytes(self, p_FirstFourChunkBytes):
        """
         COM, CPA, FIN, or ____ (four blanks).
        """
        status = str(p_FirstFourChunkBytes)
        return status


class Range_Hydrophone_Canada_2021(Range_Hydrophone_Canada):

    # New class to overwrite byte specification that Bruce changed in 2021 to include seconds since mid.

    def __init__(self,
                 p_fs = 204800,
                 p_window_size_time_time = 1.5):
        super().__init__(p_fs,p_window_size_time_time)
        self.list_time_seconds_since_midnight = []


    def _set_timestamp(self,p_seconds_since_midnight_byte):
        seconds_since_midnight_float = \
            struct.unpack('>d', p_seconds_since_midnight_byte)
        self.list_time_seconds_since_midnight.append(seconds_since_midnight_float)
        return

    def _decode_data_chunk(self, p_dataBytes):
        """
        status = decodeStatus[0:4]
        sequenceNumber = decodeSequenceNumber[4:8]
        voltageRange = decodeCardRange[8:16]
        data = decodeDataChunk[16:]
        """
        sequenceNumber = p_dataBytes[4:8]
        voltageRange = p_dataBytes[8:16]
        seconds_since_midnight = p_dataBytes[16:24]
        self._set_timestamp(seconds_since_midnight)
        byteData = p_dataBytes[24:]
        data = np.zeros([self.INTEGER_SAMPLES_PER_CHUNK], dtype='int')
        for i in range(self.INTEGER_SAMPLES_PER_CHUNK):
            try:
                data[i] = struct.unpack('>i', byteData[int(i * 4):int((i + 1) * 4)])[0]
            except:
                x = 1
        data_float = data.astype('float64') / 20e6
        try:
            status = p_dataBytes[0:4].decode("utf-8")
            return data_float, status
        except:
            return data_float, 'FAIL'


def range_binary_to_dictionary(
    p_fname_south,
    p_fname_north,
    p_binary_dir = _dirs.DIR_BINARY_HYDROPHONE,
    p_range_dictionary = _vars.RANGE_DICTIONARY):
    
    temp = dict()
    fname = p_binary_dir + p_fname_south
    hyd = \
        Range_Hydrophone_Canada(p_range_dictionary)
    hyd.load_range_specifications(p_range_dictionary)
    uncalibratedDataFloats_south, labelFinder_s, message = hyd.load_data_raw_single_hydrophone(fname)
    temp['South'] = uncalibratedDataFloats_south
    temp['South labels'] = labelFinder_s
    
    fname = p_binary_dir + p_fname_north
    hyd = \
        Range_Hydrophone_Canada(p_range_dictionary)
    hyd.load_range_specifications(p_range_dictionary)
    uncalibratedDataFloats_north, labelFinder_n, message = hyd.load_data_raw_single_hydrophone(fname)
    temp['North'] = uncalibratedDataFloats_north
    temp['North labels'] = labelFinder_n
        
    return temp


def filenames_from_df_row(p_row):
    south = p_row['South hydrophone raw']
    north = p_row['North hydrophone raw']
    return south,north


def process_single_run(
        p_df_row,
        p_dir_source = _dirs.DIR_BINARY_HYDROPHONE,
        p_dir_target = _dirs.DIR_HDF5_HYDROPHONE,
        p_dict_range_config = _vars.RANGE_DICTIONARY):
    runID = p_df_row['Run ID'].values[0]
    fname_s, fname_n = filenames_from_df_row(p_df_row)
    dict_data = range_binary_to_dictionary(
        fname_s,
        fname_n,
        p_dir_source,
        p_dict_range_config)
    fname_hdf5 = p_dir_target + runID + r'_range_hydrophone.hdf5'  
    with h5.File(fname_hdf5, 'w') as file:
        for data_type,data in dict_data.items():
            # note that not all variable types are supported but string and int are
            file[data_type] = data


def process_batch_from_df(
        p_df,
        p_dir_source,
        p_dir_target,
        p_dict_range_config = _vars.RANGE_DICTIONARY):
    
    for index,row in p_df.iterrows():    
        runID = row['Run ID']
        fname_s, fname_n = filenames_from_df_row(row)
        if fname_s =='failed' or fname_n == 'failed' : continue
        dict_data = range_binary_to_dictionary(
            fname_s,
            fname_n,
            p_dir_source,
            p_dict_range_config)
        fname_hdf5 = p_dir_target + runID + r'_range_hydrophone.hdf5'  
        with h5.File(fname_hdf5, 'w') as file:
            for data_type,data in dict_data.items():
                # note that not all variable types are supported but string and int are
                file[data_type] = data


if __name__ =='__main__':
    BATCH_RUN = True
    SINGLE_RUN = False
    if BATCH_RUN:
        local_df = pd.read_csv(_dirs.TRIAL_MAP)
        source_dir = _dirs.DIR_BINARY_HYDROPHONE
        target_dir = _dirs.DIR_HDF5_HYDROPHONE
        dict_range_config = _vars.RANGE_DICTIONARY
        if not ( os.path.isdir(target_dir)) : # need to make dir if doesnt exist
            os.mkdir(target_dir)
        process_batch_from_df(local_df,source_dir,target_dir,dict_range_config)

    if SINGLE_RUN:
        runid = r'DRJ2PB11EX00WB'
        fname_s = r'RUN_ES0451_DYN_032_000_WEST_Shyd_PORT_TM.bin'
        fname_n = r'RUN_ES0451_DYN_032_000_WEST_Nhyd_STBD_TM.bin'
        
        res = range_binary_to_dictionary(
            fname_s, fname_n)

    

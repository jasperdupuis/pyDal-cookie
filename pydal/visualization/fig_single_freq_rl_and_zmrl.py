# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:43:44 2023

@author: Jasper
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis #figsize, fontsize, etc

LOOP = False
ONE = True
TARGET_FREQ = 71
HYDRO       = 'North'
# HYDRO       = 'South'


def scatter_time_series(t,x,ax,label):
    ax.plot(t,x,marker='.',linestyle='None',label=label) 
    return ax


def plot_ambient_level_single_f(self,ax):
    """
    Adds the ambient data as horizontal lines.
    Convert to arrays
    """
    amb_nn = np.array(self.amb_n)
    select = amb_nn[:,self.target_freq_index]
    for r,s in zip(self.amb_runs,select):
        if r[:4] == 'AMJ1': ax.axhline(s,color='c')
        if r[:4] == 'AMJ2': ax.axhline(s,color='b')
        if r[:4] == 'AMJ3': ax.axhline(s,color='r')
    return ax


def load_target_spectrogram_data(p_runID,p_data_dir):
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


def extract_target_frequency(
                             p_run_data_dict,
                             p_target_index):
    samp_n = p_run_data_dict['North_Spectrogram'][p_target_index,:]
    samp_s = p_run_data_dict['South_Spectrogram'][p_target_index,:]
    t_n = p_run_data_dict['North_Spectrogram_Time'] \
        - np.min(p_run_data_dict['North_Spectrogram_Time'])
    t_s = p_run_data_dict['South_Spectrogram_Time'] \
        - np.min(p_run_data_dict['South_Spectrogram_Time'])
    return samp_n, t_n, samp_s, t_s 


def scatter_selected_data_single_f(
            p_runs,
            p_data_full_path,
            p_type, #Ambient AM or dynamic DR
            p_target_freq,
            p_day = 'DRJ3',
            p_speed = '07',
            p_hyd = 'NORTH',
            p_decibel_bool = True,
            p_ambients_bool =False):
        """
        Loops over ALL available hdf5 data looking for runs that meet passed 
        query criteria and plots a scatter at the target freq
        """
        # find the desired freq's index within the freq basis 
        target_freq = p_target_freq
        
        # The below plots the spectral time series for a selected frequency.        
        fig1,ax1 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
        fig2,ax2 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
        # Adds the run data:
        for runID in p_runs:
            # if (p_day not in runID): continue # this run is not the right day
            if (p_speed not in runID): continue # this run is not the right speed
            if 'frequency' in runID: continue # this is not a run.
            if 'summary' in runID: continue # this is not a run
            
            runData = load_target_spectrogram_data(runID,  # Returns linear values
                                                   p_data_full_path)
            target_f_index = pydal.utils.find_target_freq_index(
                target_freq,
                runData['Frequency'])
            samp_n,t_n,samp_s,t_s = \
                extract_target_frequency(runData,
                                         target_f_index)
            if p_decibel_bool:
                samp_n = 10*np.log10(samp_n / _vars.REF_UPA)
                samp_s = 10*np.log10(samp_s / _vars.REF_UPA)
                
            if p_type == 'DR' :
                #Ambient wont have this data, treat in next if
                x = runData['X'][:]
                y = runData['Y'][:]
                r = np.sqrt(x*x + y*y) 
                index_cpa = np.where(r==np.min(r))[0][0]
                
                if len(y) < len(samp_n):
                    samp_n = samp_n[:len(y)]
                    samp_s = samp_s[:len(y)]
                if len(y) > len(samp_n):
                    y = y[:len(samp_n)]
                
                samp_n_zmrl = samp_n - np.mean(samp_n)
                samp_s_zmrl = samp_s - np.mean(samp_s)
                
                
                if p_hyd == 'NORTH'.capitalize():
                    # ax = scatter_time_series(t_n, samp_n, ax, label=runID)
                    ax1 = scatter_time_series(y, samp_n, ax1, label=runID)
                    ax2 = scatter_time_series(y, samp_n_zmrl, ax2, label=runID)
                    t_n = t_n-np.min(t_n)
                    # plt.axvline( t_n [ index_cpa ] )
            
                if p_hyd == 'SOUTH'.capitalize():
                    # ax = scatter_time_series(t_s, samp_s, ax, label=runID)
                    ax1 = scatter_time_series(y, samp_s, ax1, label=runID)
                    ax2 = scatter_time_series(y, samp_s_zmrl, ax2, label=runID)
                    t_s = t_s-np.min(t_s)
                    # plt.axvline( t_s [ index_cpa ] )
                    
            if p_type =='AM' :
                if p_hyd == 'NORTH'.capitalize():
                    ax1.axhline(np.mean(samp_n))
                    # ax = scatter_time_series(t_n, samp_n, ax, label=runID)
                    # t_n = t_n-np.min(t_n)
                    
                if p_hyd == 'SOUTH'.capitalize():
                    ax1.axhline(np.mean(samp_s))
                    # ax = scatter_time_series(t_s, samp_s, ax, label=runID)
                    # t_s = t_s-np.min(t_s)
                    
        if p_ambients_bool:
            ax1 = plot_ambient_level_single_f(ax1)
        
        ax1.axvline(0)
        ax2.axvline(0)
       
        ax1.set_xlabel('Range Y-coordinate (m)')
        ax1.set_ylabel('Received level, db ref ${\mu}Pa^2$')
        
        ax2.set_xlabel('Range Y-coordinate (m)')
        ax2.set_ylabel('Zero-mean received level, db ref ${\mu}Pa^2$')
        
        
        fig1.suptitle(str(target_freq) + ' Hz received levels in 1 Hz BW, db ref ${\mu}Pa^2$')
        fig2.suptitle(str(target_freq) + ' Hz zero-mean received levels in 1 Hz BW, db ref ${\mu}Pa^2$')

        # plt.legend()
        return fig1,ax1,fig2,ax2
 
    
def scatter_selected_data_single_f_X_COORD(
            p_runs,
            p_data_full_path,
            p_type, #Ambient AM or dynamic DR
            p_target_freq,
            p_day = 'DRJ3',
            p_speed = '07',
            p_hyd = 'NORTH',
            p_decibel_bool = True,
            p_ambients_bool =False):
        """
        Loops over ALL available hdf5 data looking for runs that meet passed 
        query criteria and plots a scatter at the target freq
        """
        # find the desired freq's index within the freq basis 
        target_freq = p_target_freq
        
        # The below plots the spectral time series for a selected frequency.        
        fig1,ax1 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
        fig2,ax2 = plt.subplots(1, 1,figsize=_thesis.FIGSIZE_LARGE)     
        # Adds the run data:
        for runID in p_runs:
            # if (p_day not in runID): continue # this run is not the right day
            if (p_speed not in runID): continue # this run is not the right speed
            if 'frequency' in runID: continue # this is not a run.
            if 'summary' in runID: continue # this is not a run
            
            runData = load_target_spectrogram_data(runID,  # Returns linear values
                                                   p_data_full_path)
            target_f_index = pydal.utils.find_target_freq_index(
                target_freq,
                runData['Frequency'])
            samp_n,t_n,samp_s,t_s = \
                extract_target_frequency(runData,
                                         target_f_index)
            if p_decibel_bool:
                samp_n = 10*np.log10(samp_n / _vars.REF_UPA)
                samp_s = 10*np.log10(samp_s / _vars.REF_UPA)
                
            if p_type == 'DR' :
                #Ambient wont have this data, treat in next if
                x = runData['X'][:]
                y = runData['Y'][:]
                r = np.sqrt(x*x + y*y) 
                index_cpa = np.where(r==np.min(r))[0][0]
                
                if len(x) < len(samp_n):
                    samp_n = samp_n[:len(y)]
                    samp_s = samp_s[:len(y)]
                if len(x) > len(samp_n):
                    x = x[:len(samp_n)]
                
                
                samp_n_zmrl = samp_n - np.mean(samp_n)
                samp_s_zmrl = samp_s - np.mean(samp_s)
                
                
                if p_hyd == 'NORTH'.capitalize():
                    # ax = scatter_time_series(t_n, samp_n, ax, label=runID)
                    ax1 = scatter_time_series(x, samp_n, ax1, label=runID)
                    ax2 = scatter_time_series(x, samp_n_zmrl, ax2, label=runID)
                    t_n = t_n-np.min(t_n)
                    # plt.axvline( t_n [ index_cpa ] )
            
                if p_hyd == 'SOUTH'.capitalize():
                    # ax = scatter_time_series(t_s, samp_s, ax, label=runID)
                    ax1 = scatter_time_series(x, samp_s, ax1, label=runID)
                    ax2 = scatter_time_series(x, samp_s_zmrl, ax2, label=runID)
                    t_s = t_s-np.min(t_s)
                    # plt.axvline( t_s [ index_cpa ] )
                    
            if p_type =='AM' :
                if p_hyd == 'NORTH'.capitalize():
                    ax1.axhline(np.mean(samp_n))
                    # ax = scatter_time_series(t_n, samp_n, ax, label=runID)
                    # t_n = t_n-np.min(t_n)
                    
                if p_hyd == 'SOUTH'.capitalize():
                    ax1.axhline(np.mean(samp_s))
                    # ax = scatter_time_series(t_s, samp_s, ax, label=runID)
                    # t_s = t_s-np.min(t_s)
                    
        if p_ambients_bool:
            ax1 = plot_ambient_level_single_f(ax1)
        
        ax1.axvline(0)
        ax2.axvline(0)
       
        ax1.set_xlabel('Range X-coordinate (m)')
        ax1.set_ylabel('Received level, db ref ${\mu}Pa^2$')
        
        ax2.set_xlabel('Range X-coordinate (m)')
        ax2.set_ylabel('Zero-mean received level, db ref ${\mu}Pa^2$')
        
        
        fig1.suptitle(str(target_freq) + ' Hz received levels in 1 Hz BW, db ref ${\mu}Pa^2$')
        fig2.suptitle(str(target_freq) + ' Hz zero-mean received levels in 1 Hz BW, db ref ${\mu}Pa^2$')

        # plt.legend()
        return fig1,ax1,fig2,ax2
   
    
        
if __name__ == '__main__':
    overlap = _vars.OVERLAP
    fs = _vars.FS_HYD
    window = np.hanning( fs * _vars.T_HYD_WINDOW)
    
    dirname,runlist = pydal.utils.get_fully_qual_spec_path()
    
    full_data_path=  dirname 
    runs = pydal.utils.get_all_runs_in_dir(
        full_data_path )
    
    if ONE:
        fig_rl,ax_rl,fig_zmrl,ax_zmrl = scatter_selected_data_single_f(
            runs,
            full_data_path,
            _vars.TYPE,
            TARGET_FREQ,
            'DRJ3',
            _vars.SPEED,
            HYDRO.capitalize(),
            p_decibel_bool = True,
            p_ambients_bool = False) #Which hydrophone in use.
        
        figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_RL'
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        
        figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_ZMRL'
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
    
        plt.close('all')
    
    
        fig_rl,ax_rl,fig_zmrl,ax_zmrl = scatter_selected_data_single_f_X_COORD(
        runs,
        full_data_path,
        _vars.TYPE,
        TARGET_FREQ,
        'DRJ3',
        _vars.SPEED,
        HYDRO.capitalize(),
        p_decibel_bool = True,
        p_ambients_bool = False) #Which hydrophone in use.
    
        figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_RL_X'
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        
        figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_ZMRL_X'
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
    
        plt.close('all')
    
    
    if LOOP:
        for HYDRO in ['North','South']:
            for TARGET_FREQ in [55,551]:
                fig_rl,ax_rl,fig_zmrl,ax_zmrl = scatter_selected_data_single_f(
                    runs,
                    full_data_path,
                    _vars.TYPE,
                    TARGET_FREQ,
                    'DRJ3',
                    _vars.SPEED,
                    HYDRO.capitalize(),
                    p_decibel_bool = True,
                    p_ambients_bool = False) #Which hydrophone in use.
                
                figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_RL'
                fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.eps',
                            bbox_inches='tight',
                            format='eps',
                            dpi = _thesis.DPI)    
                fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.pdf',
                            bbox_inches='tight',
                            format='pdf',
                            dpi = _thesis.DPI)
                fig_rl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.png',
                            bbox_inches='tight',
                            format='png',
                            dpi = _thesis.DPI)
                
                figname = r'freq_'+str(TARGET_FREQ) + r'_'+HYDRO.capitalize()+'_ZMRL'
                fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.eps',
                            bbox_inches='tight',
                            format='eps',
                            dpi = _thesis.DPI)    
                fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.pdf',
                            bbox_inches='tight',
                            format='pdf',
                            dpi = _thesis.DPI)
                fig_zmrl.savefig(fname = _dirs.DIR_THESIS_IMGS \
                            + figname +'.png',
                            bbox_inches='tight',
                            format='png',
                            dpi = _thesis.DPI)
            
                plt.close('all')




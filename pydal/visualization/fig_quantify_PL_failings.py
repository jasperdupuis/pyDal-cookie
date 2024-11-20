# -*- coding: utf-8 -*-
"""


Need to quantify "why PL models aren't good for my problem":
    
    a) quantify inter-model variability (Bellhop, kraken, ram, 20logR)
        for each pair, over all freqs/range (depends on plot), 
        calculate L1, L2.
    
    b) lowering agreement as a function of: 
        frequency  
        range
        environment
    
    c) why include 500 Hz ?

    d) integrate (a) over the entire freq / distance domain and divide by 
        corresponding "length"
    

Generally:
for each model type
for each location
for each frequency presented

must compute inter-model statistic of some sort
(20logR sounds like it would work for x_i 
     (MAE and MSE?))
    
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pydal

import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal._thesis_constants as _thesis

TARGET_DIR = _dirs.DIR_RESULT_PL_QUANTIFICATION

# ENV_LIST = ['patricia_bay_north','patricia_bay_south','pekeris','emerald','ferguson']
ENV_LIST = ['patricia_bay_north','patricia_bay_south','pekeris','emerald'] # doesnt have ferguson

MODEL_LIST = ['RAM','KRAK','BELL']

FREQ_LIST_FERG      = [40,100,200,500] # Real envs cant go to 10
FREQ_LIST_PATB      = [40,100,200,300] # Real envs cant go to 10
FREQ_LIST_IDEAL     = [10,25,50,500] # deal envs can go to 10

BASIS_KEY = 'Basis'
KRAK_KEY = 'KRAKEN'
BELL_KEY = 'BELLHOP'
RAM_KEY  = 'RAM'
LOG_KEY  = '20log(R)'
KEYS_LIST = ['KRAKEN','BELLHOP','RAM','20log(R)']

def read_PL_results_file_and_interp(
        p_data_dir,
        p_location,
        p_freq):
    """
    Read, check for monotonicity, and interpolate results on to the RAM basis

    """
    if 'patricia_bay' in p_location:
        strs = p_location.split('_')
        p_location = 'patricia_bay' 
        fname_load = p_data_dir \
                + p_location \
                + r'/' + strs[2].capitalize() \
                + r'/data/' \
                + 'patbay'\
                + '_'+str(p_freq)+'.txt'

        
    else: 
        fname_load = p_data_dir \
                + p_location \
                + r'/data/' \
                + p_location\
                + '_'+str(p_freq)+'.txt'
    
    with open(fname_load, 'r') as f:
        xr = eval(f.readline().split(':')[1])
        yr = eval(f.readline().split(':')[1])
        xk = eval(f.readline().split(':')[1])
        yk = eval(f.readline().split(':')[1])
        xb = eval(f.readline().split(':')[1])
        yb = eval(f.readline().split(':')[1])
           
    # Check for monotonic increasing for interpolation reasons:
    # Reverse the x and y array for each pair if needed
    if np.all (np.diff(xr) < 0): 
        xr=xr[::-1];yr=yr[::-1]
    if np.all (np.diff(xk) < 0): 
        xk=xk[::-1];yk=yk[::-1]
    if np.all (np.diff(xb) < 0): 
        xb=xb[::-1];yb=yb[::-1]

    yk_int = np.interp(xr,xk,yk)
    yb_int = 20*np.log10(
        np.abs(
            np.interp(xr,xb,yb)
            )
        )

    return np.array(xr),np.array(yr),np.array(yk_int),np.array(yb_int)
    

def get_permutation_list(p_str,p_list):
    """
    
    Given input list of strings and single string, 
    return list without that value
    
    (list comprehension in readable callable )
    """
    permutation_list = \
        [x for x in p_list if not (x == p_str)]
    return permutation_list


"""
Initial step is to load and unify the data over the same basis function
"""
dict_envs = dict()
for location in ENV_LIST:
    freqs = []
    if location in ['emerald','pekeris']: freqs = FREQ_LIST_IDEAL
    if location in ['ferguson']: freqs = FREQ_LIST_FERG
    if location in ['patricia_bay_north','patricia_bay_south']: freqs = FREQ_LIST_PATB
    
    dict_freq = dict()
    for freq in freqs:
        dict_local = dict()
        #Get the core model data:
        if 'patricia_bay' in location:
            basis,yr,yk,yb = \
                read_PL_results_file_and_interp(
                    _dirs.DIR_RESULT_PL_MODELS_PATBAY_ONLY,
                    location,
                    freq)
        else:
            basis,yr,yk,yb = \
                read_PL_results_file_and_interp(
                    _dirs.DIR_RESULT_PL_MODELS,
                    location,
                    freq)

        dict_local['Basis']     = basis
        dict_local['RAM']       = yr
        dict_local['KRAKEN']    = yk
        dict_local['BELLHOP']   = yb
        dict_local['20log(R)']    = -20 * np.log10(basis)        

        dict_freq[freq] = dict_local
    dict_envs[location] = dict_freq


"""

PLOTS

Now that I have the results and deltas on same basis, compute L1 and plot them.
"""

for location,dict_freq in dict_envs.items():
    for freq,dict_model in dict_freq.items():
        # permutation_list = get_permutation_list(
        #     RAM_KEY,KEYS_LIST)
        
        basis   = dict_model[BASIS_KEY]
        yr      = dict_model[RAM_KEY]
        dk      = yr - dict_model[KRAK_KEY]
        db      = yr - dict_model[BELL_KEY]
        ds      = yr - dict_model[LOG_KEY]
        
        k_MAE    = np.mean(dk)
        b_MAE    = np.mean(db)
        s_MAE    = np.mean(ds)
        
        k_rMSE   = str(np.sqrt(np.mean(dk**2))).split('.')
        b_rMSE   = str(np.sqrt(np.mean(db**2))).split('.')
        s_rMSE   = str(np.sqrt(np.mean(ds**2))).split('.')
        
        k_rMSE   = k_rMSE[0]+'.'+k_rMSE[1][:1]
        b_rMSE   = b_rMSE[0]+'.'+b_rMSE[1][:1]
        s_rMSE   = s_rMSE[0]+'.'+s_rMSE[1][:1]
        
        fig,ax = plt.subplots(figsize=_thesis.FIGSIZE_QUAD)
        ax.scatter(basis,dk,marker='.',color='blue',label='KRAKEN, rMSE = '+ k_rMSE)        
        ax.scatter(basis,db,marker='.',color='orange',label='BELLHOP, rMSE = '+ b_rMSE)        
        ax.scatter(basis,ds,marker='.',color='green',label='20log(R), rMSE = '+ s_rMSE)
        
        ax.axhline(k_MAE,color='blue',linestyle='--')
        ax.axhline(b_MAE,color='orange',linestyle='--')
        ax.axhline(s_MAE,color='green',linestyle='--')        
        
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        
        ax.set_ylabel('Delta from $N_{PL,RAM}$, dB ref 1 ${\mu}Pa^2$',fontsize = _thesis.SIZE_AX_LABELS)
        ax.set_xlabel('Hydrophone distance (m)',fontsize = _thesis.SIZE_AX_LABELS)
        plt.ylim(-30,30)
        if 'patricia' in location:
            plt.xlim(100,160)
        plt.grid(which='both')
        plt.legend(loc='upper right')

        figname = location + '_' + str(freq) + '_MAE'
        plt.savefig( dpi = 300,
                    bbox_inches='tight',
                    fname = TARGET_DIR + figname + r'.png')
        plt.savefig(dpi = 300,
                    bbox_inches='tight',
                    fname = TARGET_DIR + figname + r'.pdf')
        plt.close('all')


"""

MAE / rMSE summary plots for patbay only

Now, display MAE and rMSE for both Patricia Bay hydrophones only,
for 30-300 Hz band.

"""

freqs = np.arange(30,301)
locations = ['patricia_bay_north','patricia_bay_south']

for location in locations:
    dk_arr_L1 = np.zeros_like(freqs)
    ds_arr_L1 = np.zeros_like(freqs)
    db_arr_L1 = np.zeros_like(freqs)
    
    dk_arr_L2 = np.zeros_like(freqs)
    ds_arr_L2 = np.zeros_like(freqs)
    db_arr_L2 = np.zeros_like(freqs)

    for index in range(len(freqs)):
        if freqs[index] == 44 : continue
        if freqs[index] == 74 : continue
        basis,yr,yk,yb = \
            read_PL_results_file_and_interp(
                _dirs.DIR_RESULT_PL_MODELS_PATBAY_ONLY,
                location,
                freqs[index])
        ys                  = -20 * np.log10(basis)
        
        dk = yr - yk
        db = yr - yb
        ds = yr - ys
        
        dk_arr_L1[index] = np.mean(dk)
        ds_arr_L1[index] = np.mean(db)
        db_arr_L1[index] = np.mean(ds)
        
        dk_arr_L2[index] = np.sqrt(np.mean(dk**2))
        ds_arr_L2[index] = np.sqrt(np.mean(db**2))
        db_arr_L2[index] = np.sqrt(np.mean(ds**2))
    
        
    """
    PLOT THE FREQ - MAE / MSE PLOTS
    """
    fig,ax = plt.subplots(figsize=_thesis.FIGSIZE_LARGE)
    plt.ylim(-20,20)
    plt.xlim(30,300)
    plt.xscale('log')
    plt.grid(which='both')

    ax.scatter(freqs,dk_arr_L1,marker='.',label='MAE, KRAKEN')        
    ax.scatter(freqs,db_arr_L1,marker='1',label='MAE, BELLHOP')        
    ax.scatter(freqs,ds_arr_L1,marker='v',label='MAE, 20log(R)')
    
    ax.scatter(freqs,dk_arr_L2,marker='.',label='rMSE, KRAKEN')        
    ax.scatter(freqs,db_arr_L2,marker='1',label='rMSE, BELLHOP')        
    ax.scatter(freqs,ds_arr_L2,marker='v',label='rMSE, 20log(R)')
    
    ax.set_ylabel('Delta from $N_{PL,RAM}$, dB ref 1 ${\mu}Pa^2$',fontsize = _thesis.SIZE_AX_LABELS)
    ax.set_xlabel('Frequency (Hz)',fontsize = _thesis.SIZE_AX_LABELS)
    plt.legend(loc='lower left')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    
    if 'north' in location:
        figname = location + '_' + str(freq) + '_MAE_MSE_summary_north'
    if 'south' in location:
        figname = location + '_' + str(freq) + '_MAE_MSE_summary_south'
    plt.savefig( dpi = 300,
                bbox_inches='tight',
                fname = TARGET_DIR + figname + r'.png')
    plt.savefig(dpi = 300,
                bbox_inches='tight',
                fname = TARGET_DIR + figname + r'.pdf')
    plt.close('all')
    
    """
    PLOT THE INTEGRATED MAE / MSE BARCHART
    """

    k_mae = np.mean(dk_arr_L1)
    b_mae = np.mean(db_arr_L1)
    s_mae = np.mean(ds_arr_L1)

    k_mse = np.mean(dk_arr_L2)
    b_mse = np.mean(db_arr_L2)
    s_mse = np.mean(ds_arr_L2)
    
    
    fig, ax = plt.subplots()
    bottom_labels = ['KRAKEN','BELLHOP','20log(R)']
    
    ax.bar(bottom_labels,[k_mse,b_mse,s_mse],width=0.6,bottom=0,label='rMSE')
    ax.bar(bottom_labels,[k_mae,b_mae,s_mae],width=0.6,bottom=0,label='MAE')
    
    ax.set_ylabel('Mean MAE or rMSE, dB ref 1 ${\mu}Pa^2$',fontsize = _thesis.SIZE_AX_LABELS)
    plt.legend()
    plt.grid(which='both')
    
    if 'north' in location:
        figname = location + '_' + str(freq) + '_MAE_MSE_bar_north'
    if 'south' in location:
        figname = location + '_' + str(freq) + '_MAE_MSE_bar_south'
    plt.savefig( dpi = 300,
                bbox_inches='tight',
                fname = TARGET_DIR + figname + r'.png')
    plt.savefig(dpi = 300,
                bbox_inches='tight',
                fname = TARGET_DIR + figname + r'.pdf')
    plt.close('all')
    







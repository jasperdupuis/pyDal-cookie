# -*- coding: utf-8 -*-
"""

correlation analysis in x and y coordinate

"""

from scipy import signal
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pydal.utils
import pydal.models.SLR_with_transforms

import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis

HYDRO = 'South' # Fixed, not looped over.

ANALYSIS_FREQ_LIMIT = 300

GENERATE_DATA       = False
GENERATE_DATA_2020  = True
VISUALIZE           = False
VISUALIZE_2020      = True

def get_and_sort_sample(x,y,rl_x,rl_y,length):
    """
    Make an index of (parameter) length, use it to select over the other parameters,
    then re-sort by x and y coordinate to generate separate sorted rl-x and rl-y values.
    
    Works for 1d data only.
    """    
    idx            = np.random.choice(np.arange(len(rl_x)), length, replace=False)
    x_x            = x[idx]
    y_y            = y[idx]
    rl_x           = rl_x[idx]
    rl_y           = rl_y[idx]
    # These are now sampled, but unordered due to random choice; 
    # Must re-sort:
    x_sort_ind     = np.argsort(x_x)
    y_sort_ind     = np.argsort(y_y)
    rl_xx          = rl_x[x_sort_ind]
    rl_yy          = rl_y[y_sort_ind]
    
    return x_x,y_y,rl_xx,rl_yy
    
if __name__ == "__main__":
    
    if GENERATE_DATA:
        
        data                = \
            pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
        f,rl_s,rl_n,x,y     = \
            data['Frequency'],data['South'],data['North'],data['X'],data['Y']
        run_lengths         = data['Run_Lengths']
        
        f = data['Frequency'][:ANALYSIS_FREQ_LIMIT]
        rl_nn = rl_n[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING
        rl_ss = rl_s[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING
        
        del rl_n,rl_s # not needed anymore
        
        x_sort_ind          = np.argsort(x)
        y_sort_ind          = np.argsort(y)
        
        results_x     = dict()
        results_y     = dict()
        
        for index in range(len(f)):
            temp_x      = dict()
            temp_y      = dict()
            x_x         = x[x_sort_ind]
            y_y         = y[y_sort_ind]
            if HYDRO.capitalize() == 'North':    
                rl_x    = rl_nn[index,x_sort_ind]
                rl_y    = rl_nn[index,y_sort_ind]
                # subscript denotes the sorting index
                # RL , x, y are now sorted by x or y values (lower to higher)
                
            if HYDRO.capitalize() == 'South':    
                rl_x    = rl_ss[index,x_sort_ind]
                rl_y    = rl_ss[index,y_sort_ind]
                # subscript denotes the sorting index
                # RL , x, y are now sorted by x or y values (lower to higher)        
            
            sample_size     = int ( len( rl_x ) * 0.1 )
            test_size       = int ( len( rl_x ) * 0.1 ) 
            xx,yy,rl_xx,rl_yy = get_and_sort_sample(x_x, y_y, rl_x, rl_y, sample_size)
            xx2,yy2,rl_xx2,rl_yy2 = get_and_sort_sample(x_x, y_y, rl_x, rl_y, sample_size)
            
            result_corr_x = signal.correlate(rl_xx,rl_xx2,mode='same')
            result_pearson_x = stats.pearsonr(rl_xx,rl_xx2)
            temp_x['Pearson'] = result_pearson_x
            temp_x['Correlation'] = np.mean(result_corr_x)
            
            result_corr_y = signal.correlate(rl_yy,rl_yy2,mode='same')
            result_pearson_y = stats.pearsonr(rl_yy,rl_yy2)
            temp_y['Pearson'] = result_pearson_y
            temp_y['Correlation'] = np.mean(result_corr_y)    
            
            results_x[f[index]] = temp_x 
            results_y[f[index]] = temp_y
            
            if index == 150:
                z = 1
                # Testing break point only.
            
        target_dir = _dirs.DIR_RESULT_CORRELATION
        fname_x     = 'correlations_x_'+HYDRO+'.pkl'
        fname_y     = 'correlations_y_'+HYDRO+'.pkl'
        
        pydal.utils.dump_pickle_file(results_x,target_dir,fname_x)
        pydal.utils.dump_pickle_file(results_y,target_dir,fname_y)


    if GENERATE_DATA_2020:
     
        data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(
            p_fname         =  'concatenated_data_2020.pkl'
            ) 
        f           = data2020['Frequency']
        rl_s        = data2020['South'] # 2d array, zero mean gram
        rl_n        = data2020['North'] # 2d array, zero mean gram
        rl_s        = rl_s / _vars.RL_SCALING #normalize to roughly -1/1    
        rl_n        = rl_n / _vars.RL_SCALING #normalize to roughly -1/1    
        x           = data2020['X'] / _vars.X_SCALING
        y           = data2020['Y'] / _vars.Y_SCALING
        
        f = f[:ANALYSIS_FREQ_LIMIT]
        rl_nn = rl_n[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING
        rl_ss = rl_s[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING
        
        del rl_n,rl_s # not needed anymore
        
        x_sort_ind          = np.argsort(x)
        y_sort_ind          = np.argsort(y)
        
        results_x     = dict()
        results_y     = dict()
        
        for index in range(len(f)):
            temp_x      = dict()
            temp_y      = dict()
            x_x         = x[x_sort_ind]
            y_y         = y[y_sort_ind]
            if HYDRO.capitalize() == 'North':    
                rl_x    = rl_nn[index,x_sort_ind]
                rl_y    = rl_nn[index,y_sort_ind]
                # subscript denotes the sorting index
                # RL , x, y are now sorted by x or y values (lower to higher)
                
            if HYDRO.capitalize() == 'South':    
                rl_x    = rl_ss[index,x_sort_ind]
                rl_y    = rl_ss[index,y_sort_ind]
                # subscript denotes the sorting index
                # RL , x, y are now sorted by x or y values (lower to higher)        
            
            sample_size     = int ( len( rl_x ) * 0.1 )
            test_size       = int ( len( rl_x ) * 0.1 ) 
            xx,yy,rl_xx,rl_yy = get_and_sort_sample(x_x, y_y, rl_x, rl_y, sample_size)
            xx2,yy2,rl_xx2,rl_yy2 = get_and_sort_sample(x_x, y_y, rl_x, rl_y, sample_size)
            
            result_corr_x = signal.correlate(rl_xx,rl_xx2,mode='same')
            result_pearson_x = stats.pearsonr(rl_xx,rl_xx2)
            temp_x['Pearson'] = result_pearson_x
            temp_x['Correlation'] = np.mean(result_corr_x)
            
            result_corr_y = signal.correlate(rl_yy,rl_yy2,mode='same')
            result_pearson_y = stats.pearsonr(rl_yy,rl_yy2)
            temp_y['Pearson'] = result_pearson_y
            temp_y['Correlation'] = np.mean(result_corr_y)    
            
            results_x[f[index]] = temp_x 
            results_y[f[index]] = temp_y
            
            if index == 150:
                z = 1
                # Testing break point only.
            
        target_dir = _dirs.DIR_RESULT_CORRELATION
        fname_x     = 'correlations_x_2020_'+HYDRO+'.pkl'
        fname_y     = 'correlations_y_2020_'+HYDRO+'.pkl'
        
        pydal.utils.dump_pickle_file(results_x,target_dir,fname_x)
        pydal.utils.dump_pickle_file(results_y,target_dir,fname_y)
        
       
    if VISUALIZE:
        target_dir = _dirs.DIR_RESULT_CORRELATION
        fname_x     = 'correlations_x_'+HYDRO+'.pkl'
        fname_y     = 'correlations_y_'+HYDRO+'.pkl'

        results_x = pydal.utils.load_pickle_file(target_dir,fname_x)
        results_y = pydal.utils.load_pickle_file(target_dir,fname_y)
        
        f = list( results_x.keys() )        

        corr_x,corr_y,pear_x,pear_y,p_x,p_y = \
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f))
                
        for index in range(len(f)):
            k = f[index]
            x = results_x[k]
            y = results_y[k]
            corr_x[index]  = x['Correlation']
            pear_x[index]  = x['Pearson'][0]
            p_x[index]     = x['Pearson'][1]
            corr_y[index]  = y['Correlation']
            pear_y[index]  = y['Pearson'][0]
            p_y[index]     = y['Pearson'][1]
            
        # fig,axs = plt.subplots(nrows=3,ncols=1,figsize = _thesis.FIGSIZE_TRIPLE_STACK )
        fig,axs = plt.subplots(nrows=3,ncols=1,figsize = (7,7) )

        axs[0].scatter(f,pear_x,marker='o',label='X-coordinate')
        axs[0].scatter(f,pear_y,marker='.',label='Y-coordinate')
        axs[0].set_ylabel('Pearson R' ,fontsize=_thesis.SIZE_AX_LABELS)        

        axs[1].scatter(f,p_x,marker='o',label='X-coordinate')
        axs[1].scatter(f,p_y,marker='.',label='Y-coordinate')
        axs[1].set_ylabel('Pearson p-value' ,fontsize=_thesis.SIZE_AX_LABELS)

        axs[2].scatter(f,corr_x,marker='o',label='X-coordinate')
        axs[2].scatter(f,corr_y,marker='.',label='Y-coordinate')
        axs[2].set_ylabel('Mean correlation' ,fontsize=_thesis.SIZE_AX_LABELS)
        
        fig.align_ylabels()
        plt.legend(ncols=3,bbox_to_anchor=(0.5,3.65),loc='center')
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        

        target_dir  = _dirs.DIR_RESULT_CORRELATION
        figname     =  'correlation_explanation_'+ HYDRO
        plt.savefig(fname = target_dir \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        plt.savefig(fname = target_dir \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        plt.savefig(fname = target_dir \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        plt.close('all')



   
    if VISUALIZE_2020:
        target_dir = _dirs.DIR_RESULT_CORRELATION
        fname_x     = 'correlations_x_2020_'+HYDRO+'.pkl'
        fname_y     = 'correlations_y_2020_'+HYDRO+'.pkl'

        results_x = pydal.utils.load_pickle_file(target_dir,fname_x)
        results_y = pydal.utils.load_pickle_file(target_dir,fname_y)
        
        f = list( results_x.keys() )        

        corr_x,corr_y,pear_x,pear_y,p_x,p_y = \
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f)),\
            np.zeros(len(f))
                
        for index in range(len(f)):
            k = f[index]
            x = results_x[k]
            y = results_y[k]
            corr_x[index]  = x['Correlation']
            pear_x[index]  = x['Pearson'][0]
            p_x[index]     = x['Pearson'][1]
            corr_y[index]  = y['Correlation']
            pear_y[index]  = y['Pearson'][0]
            p_y[index]     = y['Pearson'][1]
            
        # fig,axs = plt.subplots(nrows=3,ncols=1,figsize = _thesis.FIGSIZE_TRIPLE_STACK )
        fig,axs = plt.subplots(nrows=3,ncols=1,figsize = (7,7) )

        axs[0].scatter(f,pear_x,marker='o',label='X-coordinate')
        axs[0].scatter(f,pear_y,marker='.',label='Y-coordinate')
        axs[0].set_ylabel('Pearson R' ,fontsize=_thesis.SIZE_AX_LABELS)        

        axs[1].scatter(f,p_x,marker='o',label='X-coordinate')
        axs[1].scatter(f,p_y,marker='.',label='Y-coordinate')
        axs[1].set_ylabel('Pearson p-value' ,fontsize=_thesis.SIZE_AX_LABELS)

        axs[2].scatter(f,corr_x,marker='o',label='X-coordinate')
        axs[2].scatter(f,corr_y,marker='.',label='Y-coordinate')
        axs[2].set_ylabel('Mean correlation' ,fontsize=_thesis.SIZE_AX_LABELS)
        
        fig.align_ylabels()
        plt.legend(ncols=3,bbox_to_anchor=(0.5,3.65),loc='center')
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        

        target_dir  = _dirs.DIR_RESULT_CORRELATION
        figname     =  'correlation_explanation_2020_'+ HYDRO
        plt.savefig(fname = target_dir \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        plt.savefig(fname = target_dir \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        plt.savefig(fname = target_dir \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        plt.close('all')







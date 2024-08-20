# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:19:56 2024

@author: Jasper
"""

import torch
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
# import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import pydal.utils

from pydal._distribution_names import _distn_names
import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis



ANALYSIS_FREQ_LIMIT = 300


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution_obj = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution_obj.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution_obj.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x,name=distribution).plot(ax=ax)
                    
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution_obj, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


data                = \
    pydal.utils.load_training_data(p_bool_true_for_dict = True) # 
f,rl_s,rl_n,x,y     = \
    data['Frequency'],data['South'],data['North'],data['X'],data['Y']
run_lengths         = data['Run_Lengths']
x,y                 = torch.tensor(x),torch.tensor(y)

f = data['Frequency'][:ANALYSIS_FREQ_LIMIT]
rl_nn = rl_n[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING
rl_ss = rl_s[:ANALYSIS_FREQ_LIMIT,:] * _vars.RL_SCALING


res_n,res_s = np.zeros_like(rl_nn),np.zeros_like(rl_ss)


# Determine the STD of each zero-mean chunk on the y-axis
n_y_sets    = _vars.N_Y_BINS               # N segments there are
y_step      = _vars.Y_LENGTH  / n_y_sets   # how long each seg is
y_values    = ( np.arange( n_y_sets + 1 ) * y_step ) - 100  # array values from above


for index in range(len(y_values)):
    #    index = 2      # value for testing loop
    # Build a mask to get the right Y and label data    
    mask_low    = y_values[index] < data['Y'] 
    mask_high   = data['Y'] < (y_values[index] + y_step)
    mask        = np.logical_and(mask_low,mask_high)
    ss          = rl_ss[:,mask]    
    nn          = rl_nn[:,mask]
    # compute the zero mean arrays
    ss          = pydal.data_transforms.y_transform_0_mean(ss)
    nn          = pydal.data_transforms.y_transform_0_mean(nn)
    # store the zero-mean arrays for analysis after:
    res_s[:,mask]   = ss
    res_n[:,mask]   = nn
    
results = dict()


# Clean up memory:
del data,rl_s,rl_n,x,y, run_lengths, rl_nn,rl_ss


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


# Load data from statsmodels datasets
for index in range(res_s.shape[0]):
    data_s = pd.Series(res_s[index,::10])
    data_n = pd.Series(res_s[index,::10])

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data_n.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_distibutions = best_fit_distribution(data_n, 200, ax)
    results[int(f[index])] = best_distibutions
    best_dist = best_distibutions[0]

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
    ax.set_xlabel(u'Temp (°C)')
    ax.set_ylabel('Frequency')
    plt.legend()

    target      = _dirs.DIR_RESULT_NOISEBAND + 'histograms\\'
    figname     = r'histogram_' + str ( int ( f [ index ] ) ) .zfill(3)
    fullname    = target + figname
    plt.savefig(fname = fullname, dpi = _thesis.DPI)
    plt.close('all')

    # Make PDF with best params 
    pdf = make_pdf(best_dist[0], best_dist[1])
    
    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data_n.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    
    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
    dist_str = '{}({})'.format(best_dist[0].name, param_str)
    
    ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Temp. (°C)')
    ax.set_ylabel('Frequency')

    target      = _dirs.DIR_RESULT_NOISEBAND
    figname     = r'distro_bestfit_' + str(int(f[index])).zfill(3)
    fullname    = target + figname
    plt.savefig(fname = fullname+'.png', dpi = _thesis.DPI)
    plt.close('all')
    
    

pydal.utils.dump_pickle_file(results,
                             p_data_dir = _dirs.DIR_RESULT_NOISEBAND,
                             p_fname = 'north.pkl')
    
    
    
    
    
    
    
    
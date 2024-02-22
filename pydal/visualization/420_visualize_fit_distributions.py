# -*- coding: utf-8 -*-
"""

THIS FILE IS BROKEN - DOES NOT PRODUCE THE DISTRIBUTION PLOTS FOR ALL DISTROS


Examine zero mean spectral data, determine best fit distribution of the data.

SciPy has a ton of distributions! Sample some random space/frequency pairs
and then see what comes out on top.

fitting distributions to data originally lifted from stack overflow :

https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

"""


import warnings
import numpy as np
import pandas as pd
import random
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt

import pydal.models.SLR_with_transforms
import pydal._variables as _vars


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

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

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                if ax:
                    pd.Series(pdf, x).plot(ax=ax)
                    # end
                
                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
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


# load zero-mean data as used everywhere
fname2019   = r'concatenated_data_2019.pkl'
fname2020   = r'concatenated_data_2020.pkl'
data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
data        = pydal.utils.concat_dictionaries(data2019,data2020)
f_basis     = data['Frequency']

# Zero-mean the spectral time series in chunks according to 
# length of the track. As written here is ~10m.
n_y_sets    = _vars.N_Y_BINS            # N segments there are
y_step      = _vars.Y_LENGTH / n_y_sets # how long each seg is
y_values    = ( np.arange(21) * y_step ) - 100  # array values from above

"""
With the data loaded and the spacing set, wish to sample randomly pairs from:
    f
    y_pos bin
over which to do a distribution fitting. Need RL(y) data not histogram.
"""

n_entries       = 4
fmin            = 30
fmax            = 1000

random.seed(_vars.SEED)
nr = random.randrange(1000)

freqs           = np.zeros(4)
y_pos           = np.zeros_like(freqs)
for index in range(n_entries):
    freqs[index]    = random.randrange(fmin,fmax,1)
    y_pos[index]    = random.choice(y_values)

# now apply them:
for y,f in zip(y_pos,freqs):
    mask_low    = y < data['Y'] 
    mask_high   = data['Y'] < (y + y_step)
    mask        = np.logical_and(mask_low,mask_high)
    f_index     = pydal.utils.find_target_freq_index(f, f_basis)
    ss          = data['South'][f_index,mask]   
    nn          = data['North'][f_index,mask]   

    ss          = pydal.data_transforms.y_transform_0_mean_1d(ss)
    nn          = pydal.data_transforms.y_transform_0_mean_1d(nn)

    data_s           = pd.Series(ss[:].flatten())
    data_n           = pd.Series(nn[:].flatten())


# skew = st.skew(ss[:,:],axis=1)
# kurt = st.kurtosis(ss[:,:],axis=1)

# Plot south for comparison
plt.figure(figsize=(12,8))
ax = data_s.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# # Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_distibutions_s = best_fit_distribution(data_s, 200, ax)
best_dist_s = best_distibutions_s[0]
#gaussian
y, x = np.histogram(data_s, bins=200, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
distribution    = getattr(st, 'norm')
params           = distribution.fit(data_s)
arg = params[:-2]
loc = params[-2]
scale = params[-1]
# Calculate fitted PDF and error with fit in distribution
pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
sse = np.sum(np.power(y - pdf, 2.0))
gauss = (distribution, params, sse)

# Update plots
# ax.set_ylim(dataYLim)
# ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
# ax.set_xlabel(u'Temp (°C)')
# ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist_s[0], best_dist_s[1])

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data_s.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist_s[0].shapes + ', loc, scale').split(', ') if best_dist_s[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist_s[1])])
dist_str = '{}({})'.format(best_dist_s[0].name, param_str)

ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (°C)')
ax.set_ylabel('Frequency')
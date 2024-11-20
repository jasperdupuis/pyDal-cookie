# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:09:16 2022

Updated 20240919 for thesis quality images.

"""

import matplotlib.pyplot as plt
import numpy as np

import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis


ORIGINAL_THREE  = False
PATBAY          = True

LOCAL_TEXTSIZE  = 13 # To account for 0.475 scale factor in quad chart in latex.
TARGET_DIR      = _dirs.DIR_THESIS_IMGS 
HYDRO           ='North'


if ORIGINAL_THREE:
    DATA_DIR    = _dirs.DIR_RESULT_PL_MODELS
    
    dictionary = dict()
    dictionary['ferguson'] = [40,100,200,500]
    dictionary['pekeris'] = [10,25,50,500]
    dictionary['emerald'] = [10,25,50,500]
    
    
    for key,list_freq in dictionary.items():
        for FREQ in list_freq:
            figname = key + r'_' + str(FREQ)
            
            with open(DATA_DIR + key + r'/data/' + key + '_'+str(FREQ)+'.txt', 'r') as f:
                x_ram = eval(f.readline().split(':')[1])
                y_ram = eval(f.readline().split(':')[1])
                x_krak = eval(f.readline().split(':')[1])
                y_krak = eval(f.readline().split(':')[1])
                x_bell = eval(f.readline().split(':')[1])
                y_bell = eval(f.readline().split(':')[1])
                    
            
            # Generate plot
            fig,ax = plt.subplots(figsize=_thesis.FIGSIZE_QUAD)
            ax.plot(x_bell,20*np.log10(np.abs(y_bell)),label='BELLHOP')
            ax.plot(x_krak,y_krak,label='KRAKEN')
            ax.plot(x_ram,y_ram,label='RAM')
            ax.plot(x_bell,-20*np.log10(x_bell),label='20logR')
            # # Not needed for thesis images due to captions:
            # if key =='ferguson':
            #     fig.suptitle('Ferguson Cove, ' + str(FREQ) + ' Hz')
            # if key =='emerald_basin':
            #     fig.suptitle('Emerald Basin, ' + str(FREQ) + ' Hz')
            # if key =='pekeris':
            #     fig.suptitle('Pekeris Waveguide, ' + str(FREQ) + ' Hz')
            ax.set_ylabel('Propagation loss, dB ref 1 ${\mu}Pa^2$',fontsize = LOCAL_TEXTSIZE)
            ax.set_xlabel('Hydrophone distance (m)',fontsize = LOCAL_TEXTSIZE)
            plt.grid(which='both')
            plt.ylim(-80,-20)
            plt.legend()
            plt.savefig( dpi = 300,
                        bbox_inches='tight',
                        fname = TARGET_DIR + key + '_'+str(FREQ)+r'.png')
            plt.savefig(dpi = 300,
                        bbox_inches='tight',
                        fname = TARGET_DIR + key + '_'+str(FREQ) +r'.pdf')
            plt.close('all')
            
            
            
if PATBAY:
    DATA_DIR    = r'C:\Users\Jasper\Documents\Repo\pyDal\UWAEnvTools\results\\'
    
    dictionary = dict()
    dictionary['patbay'] = [40,100,200,300]
    
    
    for key,list_freq in dictionary.items():
        for FREQ in list_freq:
            figname = key + r'_' + str(FREQ)
            
            with open(DATA_DIR + r'patricia_bay\\' + HYDRO + r'\\data\\' + r'\\' + key + '_'+str(FREQ)+'.txt', 'r') as f:
                x_ram = eval(f.readline().split(':')[1])
                y_ram = eval(f.readline().split(':')[1])
                x_krak = eval(f.readline().split(':')[1])
                y_krak = eval(f.readline().split(':')[1])
                x_bell = eval(f.readline().split(':')[1])
                y_bell = eval(f.readline().split(':')[1])
            
            # Generate plot
            fig,ax = plt.subplots(figsize=_thesis.FIGSIZE_QUAD)
            ax.plot(x_bell,20*np.log10(np.abs(y_bell)),label='BELLHOP')
            ax.plot(x_krak,y_krak,label='KRAKEN')
            ax.plot(x_ram,y_ram,label='RAM')
            ax.plot(x_bell,-20*np.log10(x_bell),label='20logR')
            # # Not needed for thesis images due to captions:
            # if key =='ferguson':
            #     fig.suptitle('Ferguson Cove, ' + str(FREQ) + ' Hz')
            # if key =='emerald_basin':
            #     fig.suptitle('Emerald Basin, ' + str(FREQ) + ' Hz')
            # if key =='pekeris':
            #     fig.suptitle('Pekeris Waveguide, ' + str(FREQ) + ' Hz')
            ax.set_ylabel('Propagation loss, dB ref 1 ${\mu}Pa^2$',fontsize = LOCAL_TEXTSIZE)
            ax.set_xlabel('Hydrophone distance (m)',fontsize = LOCAL_TEXTSIZE)
            plt.ylim(-80,-20)
            plt.grid(which='both')
            plt.legend()
            plt.savefig( dpi = 300,
                        bbox_inches='tight',
                        fname = TARGET_DIR + 'patricia_'+str(FREQ)+r'.png')
            plt.savefig(dpi = 300,
                        bbox_inches='tight',
                        fname = TARGET_DIR + 'patricia_'+str(FREQ) +r'.pdf')
            plt.close('all')
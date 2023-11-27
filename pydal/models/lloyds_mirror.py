# -*- coding: utf-8 -*-

"""
Created on Fri Oct 13 16:04:26 2023

@author: Jasper
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Lloyds mirror in deep water, from pages ~ 20 in Jensen
"""
z_s = 2.4
z_r = 20
c = 1500

# COA page 20
# Do it for a range of Rs and a fixed f first.
f       = 5000
lamb    = c/f
k       = 2 * np.pi / lamb
R       =np.arange(1,100000)
p_mag   = (2 / R) * np.abs ( np.sin ( k * z_s * z_r / R  ) )
p2      = 10*np.log10(p_mag**2)
plt.figure();plt.plot(R,p2);plt.xscale('log');plt.xlabel('Distance (m)');plt.title('COA page 20 LME' + str(f) + ' Hz')

#Venditts ten challenges to measure URN paper
# TL = 20 log ( sin ( kd sin ( theta ) ) ) + 20 log r
theta   = np.arctan(z_s / R)
k       = 2 * np.pi / lamb
TL      = 20 * np.log10 ( np.sin( k * z_s * np.sin(theta))) + 20*np.log10(R)
plt.figure();plt.plot(R,TL);plt.xscale('log');plt.xlabel('Distance (m)');plt.title('Vendittis LME,' + str(f) + ' Hz')


# COA page 20
# Do it for a range of Rs and a fixed f first.
f       = np.arange(1,10000)
lamb    = c/f
k       = 2 * np.pi / lamb
R       = 100
p_mag   = (2 / R) * np.abs ( np.sin ( k * z_s * z_r / R  ) )
p2      = 10*np.log10(p_mag**2)
plt.figure();plt.plot(f,p2);plt.xscale('log');plt.xlabel('Frequency (Hz)');plt.title('COA page 20 LME, ' + str(R) + ' m')

#Venditts ten challenges to measure URN paper
theta   = np.arctan(z_s / R)
k       = 2 * np.pi / lamb
TL      = 20 * np.log10 ( np.sin( k * z_s * np.sin(theta))) + 20*np.log10(R)
plt.figure();plt.plot(R,TL);plt.xscale('log');plt.xlabel('Frequency (Hz)');plt.title('Vendittis LME,' + str(R) + ' m')
    


"""
Look at a gram for a run i choose at random from the run_list variable.
"""
import pydal.utils

dir_spec ,run_list  = pydal.utils.get_fully_qual_spec_path()
p_runID             = run_list[16]
gram_dict,N         = pydal.utils.get_spectrogram_file_as_dict(p_runID, dir_spec)    

n                   = gram_dict['North_Spectrogram'][:50000,:]
ndb                 = 10 * np.log10 (n) 
s                   = gram_dict['South_Spectrogram'][:50000,:]
sdb                 = 10 * np.log10 (s) 


plt.figure();plt.imshow(ndb,aspect='auto',origin='lower');plt.title('North Gram');plt.colorbar()
plt.figure();plt.imshow(sdb,aspect='auto',origin='lower');plt.title('South Gram');plt.colorbar()


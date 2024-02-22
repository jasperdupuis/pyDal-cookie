# -*- coding: utf-8 -*-
"""

This is the file to send Adam the images in email in Jan 2024.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir_2020results = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\data\raw\2020-Orca Ranging\Pat Bay Data\ES0453_MOOSE_SPC_DYN\\'

fname_pd = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal-cookie/data/raw/keel_analysis.csv'

keel_runs = [ 'DRF2PB05AA01EN',
'DRF2PB05AA01WN',
'DRF2PB07AA00EN',
'DRF2PB07AA00WN',
'DRF2PB05AA00ES']

comp_runs = dict()
df = pd.read_csv(fname_pd)

# list of runs for each keel run we did for easy comparison. Will be overlap.
for r in keel_runs:
    subsel = df[df['Run ID'].str.contains(r[4:-4])]
    runs = list(subsel['Run ID'])
    runs = [x for x in runs if not (x[-1]=='N' or x[-1] =='S')]
    comp_runs[r] = runs
    

# Selection of keel run.
keel_run    = keel_runs[0]
beam_runs   = comp_runs[keel_run]

# Get keel run fully specified OTO path
fname_keel_oto_s = \
    dir_2020results \
    + df [ df [ 'Run ID' ] ==  keel_run]['South hydrophone OTO'].values[0]
fname_keel_oto_n = \
    dir_2020results \
    + df [ df [ 'Run ID' ] ==  keel_run]['North hydrophone OTO'].values[0]

# Extract keel run range result
df_keel_s   = pd.read_csv(fname_keel_oto_s,skiprows=73)
f_k_s       = df_keel_s [ df_keel_s.columns[0]].values
v_k_s       = df_keel_s [ df_keel_s.columns[1]].values

df_keel_n   = pd.read_csv(fname_keel_oto_n,skiprows=73)
v_k_n       = df_keel_n [ df_keel_n.columns[1]].values

# 
beams_s = []
beams_n = []
for b in beam_runs:
    # Get beam run fully specified path, south only.
    fname_beam_oto_s = \
        dir_2020results \
        + df [ df [ 'Run ID' ] ==  b]['South hydrophone OTO'].values[0]
    fname_beam_oto_n = \
        dir_2020results \
        + df [ df [ 'Run ID' ] ==  b]['North hydrophone OTO'].values[0]

    if 'failed' in fname_beam_oto_s:
        continue
    # Extract keel run range result
    df_beam_s   = pd.read_csv(fname_beam_oto_s,skiprows=73)
    f_b_s       = df_beam_s    [ df_beam_s.columns[0]].values
    v_b_s       = df_beam_s    [ df_beam_s.columns[1]].values
  
    df_beam_n   = pd.read_csv(fname_beam_oto_n,skiprows=73)
    f_b_n       = df_beam_n    [ df_beam_n.columns[0]].values
    v_b_n       = df_beam_n    [ df_beam_n.columns[1]].values

    beams_s.append(v_b_s)
    beams_n.append(v_b_n)

fig,ax = plt.subplots(1,1,figsize=(12,8))
ax.step(f_k_s,v_k_s,linestyle='dashed',label=keel_run)
ax.step(f_k_s,np.mean(np.array(beams_s),axis=0),label='South: ' + str(beam_runs))
ax.step(f_k_s,np.mean(np.array(beams_n),axis=0),label='North: ' + str(beam_runs))
# for index in range(len(beams)):
#     ax.step(f_k_s,beams[index],label=beam_runs[index])
plt.xscale('log')
plt.legend()


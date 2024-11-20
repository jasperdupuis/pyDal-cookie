# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:53:30 2024

@author: Jasper
"""

import os
import cv2

import pydal._directories_and_files as _dirs

dir_full = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal\models\saved_models_1d_single_f\hdf5_spectrogram_bw_1.0_overlap_90\North\Y\high capacity\figs\\'
fnames = os.listdir(dir_full)

img_array = []
for f in fnames:
    if int((f.split('.')[0])) > 300 : continue
    fname                   = dir_full+f
    img                     = cv2.imread(fname)
    height, width, layers   = img.shape
    size = (width,height)
    img_array.append(img)
    
i = img_array[0]

fname = _dirs.DIR_RESULT + r'north_high_capacity.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter(fname, codec, 10, (width, height))

for img in img_array:
    vid_writer.write(img)

vid_writer.release()

# out = cv2.VideoWriter(fname , cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
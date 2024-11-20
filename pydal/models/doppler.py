# -*- coding: utf-8 -*-
"""

doppler

"""
import numpy as np

fs = np.arange(30,300)

c = 1500
v_s = 10
v_h = 0

fo = fs * (c + v_h) / ( c + v_s)

plt.plot(fs,fo)




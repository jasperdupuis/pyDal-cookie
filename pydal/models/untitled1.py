# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:39:59 2024

@author: Jasper
"""



fs = np.arange(1,500)
c= 1500
v = 10

fo = fs * (c / (c + v))


plt.plot(fs,fo-fs)


# -*- coding: utf-8 -*-
"""

Build a 2d 200m by 200m surface on which to test ML and COA models, with
arbitrary resolution parameter.

20230605 need to have a pytorch.nn model() alraedy trained before running this.


Created on Mon Jun  5 16:08:23 2023

@author: Jasper
"""

import numpy as np


xmax = 100
ymax = 100
dx = 0.5
dy = 0.5

x = np.arange(start = -1 * xmax, stop = xmax, step=dx)
y = np.arange(start = -1 * ymax, stop = ymax, step=dy)

xl = len(x)
yl = len(y)

xres = np.zeros(len(x) * len(y))
yres = np.zeros(len(x) * len(y))
xc = 0
for yi in range ( yl ):
    for xi in range ( xl ):
        xres[ (xc * xl) + xi ] = x[ xi ]
        yres[ (xc * yl) + xi ] = y[ yi ]
    xc = xc + 1

# test with model from elsewhere.
results = np.zeros_like(xres)
index = 0
for xf,yf in zip(xres,yres):
    test = torch.tensor((yf,xf))
    # test = test.cuda()
    res = model(test.float())
    results[index] = res
    index = index + 1

#sanity:
test = torch.tensor((2,0))
model(test.float())    

r = np.array(results)
r[1]        

result = np.reshape(results,(len(x),len(y)), order='C')
# delta = label-result

plt.imshow(result)



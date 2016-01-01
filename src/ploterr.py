#! /usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

#fname = "errors_1.dat"
fname = "../data/errors_unlim_withoutboun.dat"

data = np.genfromtxt(fname)
n = data.shape[0]

'''psigy = 0.0; sigx2 = 0.0; sigx = 0.0; psigxy = 0.0
for i in range(n):
    psigy += data[i,1]
    sigx += data[i,0]
    sigx2 += data[i,0]*data[i,0]
    psigxy += data[i,1]*data[i,0] '''

psigy = data[:,1].sum()
sigx = data[:,0].sum()
sigx2 = (data[:,0]*data[:,0]).sum()
psigxy = (data[:,1]*data[:,0]).sum()

pslope = (n*psigxy-sigx*psigy)/(n*sigx2-sigx**2)

plt.plot(data[:,0],data[:,1],'o-')
plt.title("Grid-refinement Study- " + fname + ", slope = "+str(pslope))
plt.xlabel("Log mesh size")
plt.ylabel("Log l2 error")
plt.show()

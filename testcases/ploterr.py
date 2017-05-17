#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)
	
fname = sys.argv[1]
title = fname.split('/')[-1]

data = np.genfromtxt(fname)
n = data.shape[0]

"""psigy = data[:,1].sum()
sigx = data[:,0].sum()
sigx2 = (data[:,0]*data[:,0]).sum()
psigxy = (data[:,1]*data[:,0]).sum()
pslope = (n*psigxy-sigx*psigy)/(n*sigx2-sigx**2)"""

pslope = (data[-1,1]-data[-2,1])/(data[-1,0]-data[-2,0])

print("Slope is " + str(pslope))

plt.plot(data[:,0],data[:,1],'o-')
plt.title("Grid-refinement - " + title + ", slope = "+str(pslope))
plt.xlabel("Log mesh size")
plt.ylabel("Log l2 error")
plt.show()

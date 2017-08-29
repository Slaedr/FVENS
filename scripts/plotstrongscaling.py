#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

# column to plot
col = 2

# Labels for legend - change if necessary
labels = [
			"SGS", 
			"Block-SGS", 
			"ILU0", 
			"Block-ILU0"
		]

if(len(sys.argv) < 2):
	print("Error! Please provide at least 1 input file name.")
	sys.exit(-1)

nfiles = len(sys.argv)-1
symbs = ['bo-', 'gs-', 'r^-', 'cv-','b*-']
	
for ifile in range(nfiles):
	fname = sys.argv[ifile+1]

	data = np.genfromtxt(fname)
	n = data.shape[0]

	effs = data[0,col]/(data[:,1]*data[:,col])*100.0

	plt.plot(data[:,1],effs,symbs[ifile], label=labels[ifile])

plt.title("Strong scaling efficiency")
plt.xlabel("Number of threads")
plt.ylabel("Scaling percent")
plt.legend()
plt.grid('on')
plt.show()

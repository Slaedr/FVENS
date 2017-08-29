#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

# column to plot
col = 2

# Labels to use in the legend; change as required
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

	plt.plot(data[:,1],data[:,col],symbs[ifile], label=labels[ifile])

plt.title("Timing comparison")
plt.xlabel("Number of threads")
plt.ylabel("Wall-clock time")
plt.legend()
plt.grid('on')
plt.show()

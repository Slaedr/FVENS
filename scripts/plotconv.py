""" Plots convergence histories.
	Make sure to put the description you need in the legend at the end of filenames, 
	before extensions,
	separated from whatever comes before by '-'.
	E.g.: 'In grid4-HLLflux-3wp-SGS_solver.dat', the legend label would be 'SGS_solver'.
"""

#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)

symbs = ['b-', 'g-', 'r-', 'c-','b-']
titles = []
	
for ifile in range(len(sys.argv)-1):
	fname = sys.argv[ifile+1]
	
	# For use in legend, strip prefix and extension(s)
	tstr = fname.split('/')[-1]
	titles.append(tstr.split('.')[0].split('-')[-1])

	data = np.genfromtxt(fname)
	n = data.shape[0]
	plt.plot(data[:,0],np.log10(data[:,1]),symbs[ifile], label=titles[-1])
	
titlestr = ""
for i in range(len(titles)):
	if i==0:
		titlestr = titlestr + titles[i]
	else:
		titlestr = titlestr + ", " + titles[i]

plt.title("Convergence - " + titlestr)
plt.xlabel("Pseudo-time step")
plt.ylabel("Log l2 residual")
plt.grid('on')
plt.legend()
plt.show()

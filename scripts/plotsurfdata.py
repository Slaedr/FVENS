""" Plots surface data
"""

#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)

symbs = ['b.', 'gs', 'r^', 'cv','b*']
names = ["Cp", "Cf", "other"]

plotaxis = 0

for ifile in range(len(sys.argv)-1):
    fname = sys.argv[ifile+1]
    data = np.genfromtxt(fname)
    m,n = data.shape

    title = fname.split('/')[-1]
    title = '-'.join(title.split('.')[0].split('-')[1:])

    for j in range(n-2):
        thistitle = title + "-" + names[j]
        plt.plot(data[:,plotaxis],data[:,j+2],symbs[ifile], label=thistitle)
        plt.title(names[j])
        plt.xlabel("Coordinate " + str(plotaxis))
        plt.ylabel(names[j])
        plt.grid('on')
        #plt.legend()
        plt.show()


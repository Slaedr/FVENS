import sys
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pytplot as plt
from couettesolution import *

# Constants

rho = 1.1839
Tinf = 298.0
ssound = 346.13
k = 0.026
mu = 1.84e-5

Pr = 0.71102553846
Reinf = 3217119.5652

# VTU file details
solnindex = 0

tempindex = 4
velindex = 3

coordindex = 1

# Start

couette = Couette(1.0,mu,k,50.0,55.0,Tinf,Tinf)

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)

symbs = ['bo-', 'gs-', 'r^-', 'cv-','b*-']

for ifile in range(len(sys.argv)-1):
	fname = sys.argv[ifile+1]
	title = fname.split('/')[-1]
	title = '-'.join(title.split('.')[0].split('-')[1:])
	
	# parse VTU
	tree = ET.parse(fname)
	root = tree.getroot()
	rawtemp = root[0][0][solnindex][tempindex].text
	rawvel = root[0][0][solnindex][velindex].text
	rawcoords = root[0][0][coordindex][0].text
	
	tempr = np.genfromtxt(rawtemp)
	n = tempr.shape[0]
	vel = np.genfromtxt(rawvel)
	if n != vel.shape[0]:
		print("!! Shape mismatch in vel!")
	coords = np.genfromtxt(rawcoords)
	if n != coords.shape[0]:
		print("!! Shape mismatch in coords!")

	# TODO: Compute error
	
	pslope = (data[1:,1]-data[:-1,1])/(data[1:,0]-data[:-1,0])
	for i in range(n-1):
		print("Slope "+ str(i) +" for " + title + " is " + str(pslope[i]))

	plt.plot(data[:,0],data[:,1],symbs[ifile], label=title+", slope={:.2f}".format(pslope[-1]))

plt.title("Grid-refinement")
plt.xlabel("Log mesh size")
plt.ylabel("Log l2 error")
plt.grid('on')
plt.legend()
plt.show()

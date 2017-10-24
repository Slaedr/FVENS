# Compare a computed solution with the exact solution of circular Couette flow

import sys
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from circCouetteExactSolution import *

## User input...

ri = 0.5
ro = 1.0
vi = 1.0
Ti = 1.0
mu = 2e-5

Pr = 0.72
g = 1.4
R = 287.06 # for dry air

# Normalization constants
Tinf = Ti
vinf = vi

## User input ends

Cp = g*R/(g-1.0)
k = mu*Cp/Pr
wi = vi/ri

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)

symbs = ['bo-', 'gs-', 'r^-', 'cv-','b*-']

nfile = len(sys.argv)-1
temperr = np.zeros(nfile)
vxerr = np.zeros(nfile)
h = np.zeros(nfile)

exsol = CircCouetteSolution(ri, ro, wi, Ti, mu, k)

for ifile in range(nfile):	
	fname = sys.argv[ifile+1]
	data = np.genfromtxt(fname)
	nelem,nvar = data.shape

	texact = exsol.getTemperature(data[:,0],data[:,1])/Tinf
	vxexact= exsol.getXVelocity(data[:,0],data[:,1])/vinf

	h[ifile] = np.log10(np.sqrt(1.0/nelem))
	temperr[ifile] = np.log10(np.linalg.norm(texact-data[:,6])/nelem)
	vxerr[ifile] = np.log10(np.linalg.norm(vxexact-data[:,3])/nelem)

# Orders
slopet = np.zeros(nfile-1)
slopev = np.zeros(nfile-1)
for i in range(1,nfile):
	slopet[i-1] = (temperr[i]-temperr[i-1])/(h[i]-h[i-1])
	slopev[i-1] = (vxerr[i]-vxerr[i-1])/(h[i]-h[i-1])

print("Temp. slopes " + str(slopet))
print("Velocity slopes " + str(slopev))

# Plot order
plt.plot(h, temperr, symbs[0], label = "Temp. order "+str(slopet[-1]))
plt.plot(h, vxerr, symbs[1], label = "Vel. order "+str(slopev[-1]))
plt.title("Grid-refinement")
plt.xlabel("Log mesh size")
plt.ylabel("Log l2 error")
plt.grid('on')
plt.legend()
plt.show()


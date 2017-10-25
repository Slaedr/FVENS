# Plots non-dim x-velocity and skin-friction coefficient from numerical solution and Blasius soln
# Usage: The order of cmd-line args is as follows - file names of
# 1. Blasius velocity solution
# 2. Numerical solution of velocities
# 3. Numerical solution of Cp, Cf

import sys
import numpy as np
from matplotlib import pyplot as plt

# User input

Reinf = 8.7e5
rho = 1.2
l = 1.0
Minf = 0.2
Tinf = 290.19

g = 1.4
R = 287.06

# End user input

vinf = Minf*np.sqrt(g*R*Tinf)
nu = vinf*l/Reinf
print("Free velocity = " + str(vinf) + ", kin. visc. = " + str(nu))

if(len(sys.argv) < 4):
	print("Error. Please provide input file names.")
	sys.exit(-1)

symbs = ['bo', 'gs', 'r^', 'cv','b*']

blvel = np.genfromtxt(sys.argv[1])
vel = np.genfromtxt(sys.argv[2])
cf = np.genfromtxt(sys.argv[3])

# Velocities

eta = vel[:,1]*np.sqrt(vinf/(nu*vel[:,0]))

plt.plot(vel[:,2], eta, symbs[0], label = "Computed")
plt.plot(blvel[:,1], blvel[:,0], symbs[1], label = "Blasius")
plt.title("X-velocity")
plt.xlabel("$u/u_\infty$")
plt.ylabel("$\eta$")
plt.grid('on')
plt.legend()
plt.show()

plt.plot(np.abs(vel[:,3])*np.sqrt(Reinf), eta, symbs[0], label = "Computed")
plt.plot(blvel[:,2], blvel[:,0], symbs[1], label = "Blasius")
plt.title("Y-velocity")
plt.xlabel("$v/v_\infty$")
plt.ylabel("$\eta$")
plt.grid('on')
plt.legend()
plt.show()

# Skin friction

# Blasius solution. NOTE: we assume the plate starts at x = 0
blcf = 0.664/np.sqrt(vinf*cf[:,0]/nu)

#plt.plot(np.log10(cf[:,0]),np.log10(cf[:,3]), symbs[0], label="Computed")
#plt.plot(np.log10(cf[:,0]),np.log10(blcf), symbs[1], label="Blasius")
plt.plot(cf[:,0],cf[:,3], symbs[0], label="Computed")
plt.plot(cf[:,0],blcf, symbs[1], label="Blasius")
plt.title("Skin friction coefficient")
plt.xlabel("log x")
plt.ylabel("log $C_f$")
plt.grid('on')
plt.legend()
plt.show()



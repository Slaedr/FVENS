# Plots non-dim x-velocity and skin-friction coefficient from numerical solution and Blasius soln
# Usage: The order of cmd-line args is as follows - file names of
# 1. Blasius velocity solution
# 2. Basius skin-friction coefficient
# 3. Numerical solution of velocities
# 4. Numerical solution of Cp, Cf

import sys
import numpy as np
from matplotlib import pyplot as plt

# User input

Reinf = 4269137.684
Pr = 0.712
rho = 1.2
l = 1.0
Minf = 0.2
Tinf = 297.62

g = 1.4
R = 287.06

# End user input

vinf = Minf*np.sqrt(g*R*Tinf)
nu = vinf*l/Reinf
print("Free velocity = " + str(vinf) + ", kin. visc. = " + str(nu))

if(len(sys.argv) < 5):
	print("Error. Please provide input file names.")
	sys.exit(-1)

symbs = ['bo-', 'gs-', 'r^-', 'cv-','b*-']

blvel = np.genfromtxt(sys.argv[1])
blcf = np.genfromtxt(sys.argv[2])
vel = np.genfromtxt(sys.argv[3])
cf = np.genfromtxt(sys.argv[4])

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

# Skin friction

plt.plot(np.log10(cf[:,0]),np.log10(cf[:,3]), symbs[0], label="Computed")
plt.plot(np.log10(blcf[:,0]),np.log10(blcf[:,1]), symbs[1], label="Blasius")
plt.title("Skin friction coefficient")
plt.xlabel("log x")
plt.ylabel("log $C_f$")
plt.grid('on')
plt.legend()
plt.show()



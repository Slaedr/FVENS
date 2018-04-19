#! /bin/env python3

import numpy as np

class IsentropicVortex:
    """ Isentropic vortex convected with free-stream velocity in the x-direction.
    Variables are assumed non-dimensionalized using free-stream density, free-stream velocity magnitude
    and free-stream temperature as reference variables.

    Ref: Seth Spiegel, H.T. Huynh and James R. DeBonis. "A survey of the isentropic Euler vortex problem
    using high-order methods", AIAA paper.
    """

    def __init__(self, M_inf, centre, strength, gridlen, stddev=1.0):
        """ Set the free-stream Mach number, centre of the vortex, its strenth, 
        characteristic dimension of the domain and the 'spread' of the vortex.
        """
        self.Rgas = 287.15
        self.gamma = 1.4
        self.dim = 2
        self.Minf = M_inf
        self.pinf = 1.0/(gamma*self.Minf*self.Minf)
        self.c = x_centre.copy()
        self.beta = strength
        self.R = gridlen
        self.sigma = stddev

    def omega(r):
        f = -0.5/(sigma*sigma*R*R)*(r[:,0]*r[:,0]+r[:,1]*r[:,1])
        return beta * np.exp(f)

    def velocity(r):
        # The non-dimensional free-stream velocity is (1,0)
        v = np.array(r.shape)
        mag = self.omega(r)/R
        v[:,0] = 1 + mag[:]*(-r[:,1])
        v[:,1] = mag[:] * r[:,0]
        return v

    def temperature(r):
        # Non-dimensional free-stream temperature is 1.0
        return -(self.gamma-1)/2.0 * self.omega(r)*self.omega(r)

    def writeInitialCondition(meshfilename, icfilename):
        pass


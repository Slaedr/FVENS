
import numpy as np

class CircCouetteSolution:
	""" Computation of exact solution of the circular Couette problem
	    as described by Zwanenburg and Nadarajah (AIAA 2017)
	"""
	def __init__(self, ri, ro, wi, Ti, mu, k):
		""" @param ri Inner radius
			@param ro Outer radius
			@param wi Inner angular velocity
			@param Ti Inner temperature
			@param mu Dynamic viscosity
			@param k Thermal conductivity
		"""
		self.ri = ri
		self.ro = ro
		self.wi = wi
		self.Ti = Ti
		self.mu = mu
		self.k = k
		
		self.C = wi/(1.0/(self.ri*self.ri) - 1.0/(self.ro*self.ro))
	
	def getTemperature(self, x, y):
		""" Get exact temperature
			@param x Array of x-coords
			@param y Array of y-coords
		"""
		r = np.sqrt(x*x + y*y)
		term = 2.0/(self.ro*self.ro)*np.log(r/self.ri) + 1.0/(r*r)-1.0/(self.ri*self.ri)
		return self.Ti - self.C*self.C*self.mu/self.k * term

	def getXVelocity(self, x, y):
		""" Get exact x-velocity
			@param x Array of x-coords
			@param y Array of y-coords
		"""
		r = np.sqrt(x*x + y*y)
		vtheta = self.C*r*(1.0/(r*r)-1.0/(self.ro*self.ro))
		theta = np.arctan2(y,x)
		return vtheta*(-np.sin(theta))

	def getYVelocity(self, x, y):
		""" Get exact y-velocity
			@param x Array of x-coords
			@param y Array of y-coords
		"""
		r = np.sqrt(x*x + y*y)
		vtheta = self.C*r*(1.0/(r*r)-1.0/(self.ro*self.ro))
		theta = np.arctan2(y,x)
		return vtheta*np.cos(theta)



class Couette:
	""" Exact solution of Couette flow """
	
	def __init__(self,mu,k,L,v1,v2,t1,t2):
		""" Set constants and parameters """
		self.L = L
		self.v1 = v1
		self.v2 = v2
		self.t1 = t1
		self.t2 = t2
		self.mu = mu
		self.k = k
		
		self.dv = v2-v1
		self.dt = t2-t1
	
	def velocity(self,y):
		return self.dv/self.L * y + self.v1
	
	def temperature(self,y):
		return -self.mu*self.dv**2 * y**2 /(2.0*self.k*self.L**2) + (self.dt/self.L+self.mu*self.dv**2/(2.0*self.k*self.L)) * y \
		 + self.t1

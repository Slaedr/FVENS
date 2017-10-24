""" 
@file testcase.py
@brief A framework for defining testcases
@author Aditya Kashi

  This file is part of FVENS.
    FVENS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
 
    FVENS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
"""

class ExactSolution:
	""" An abstract exact solution """
	def __init__(self, dimension):
		## The number of spatial dimensions of interest:
		self.dim = dimension

		assert(dimension == 1 or dimension == 2 or dimension == 3)

	def setup():
		pass

	def evaluate(self, p):
	""" Evaluate the exact solution at a list of points.
	@param p A numpy array containing coordinates of a list of points
	"""
	pass

class TestCase:
	""" An abstract testcase. Does not specify much currently. """
	def __init__(self, exactsol):
		## The exact solution context:
		self.exsol = exactsol
		## Number of spatial dimensions of interest:
		self.dim = exactsol.dim

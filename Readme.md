FVENS
=====

This is a cell-centered finite volume solver for the two-dimensional compressible Euler equations. Unstructured grids having both triangles and quadrangles are supported. It includes MUSCL (variable extrapolation) reconstruction using either Green-Gauss or weighted least-squares methods. WENO (weighted essentially non-oscillatory) and Venkatakrishnan limiters are available for flows with shocks. A number of numerical convective fluxes are available - local Lax-Friedrichs (Rusanov), Van Leer flux vector splitting, HLL (Harten - Lax - Van Leer), HLLC and Roe-Pike. Modified average gradients are used for viscous fluxes. 

Currently, only steady-state problems are supported. Both explicit and implicit pseudo-time stepping are avaible. Explicit time-stepping uses the forward Euler scheme while implicit time stepping uses the backward Euler scheme; both use local time-steps. A number of linear solvers and preconditioners are available.

Building
--------
The following libraries are required:
- The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix library (version 3.3.4) - needs an environment variable called EIGEN_DIR to be set to the top-level Eigen directory
- [BLASTed](https://github.com/Slaedr/BLASTed) sparse linear algebra library - needs an environment variable called BLASTED_DIR to be set to the top level BLASTed directory.

Optionally, OpenMP can be used if available (default builds of GCC on most GNU/Linux distributions have this, for instance).

To build, issue

		cmake /path/to/src -DCMAKE_BUILD_TYPE=Debug -DOMP=1

and for a release build

		cmake /path/to/src -DCMAKE_BUILD_TYPE=Release -DOMP=1 -DSSE=1

'-DOMP=1' should be removed if OpenMP is not available. Doing so will cause warnings about unknown pragmas to appear while building, which are to be ignored. Finally,

		make -j<N>

where '\<N\>' should be replaced by the number of threads to use for a parallel build.

To build the Doxygen documentation, please type the following command in the doc/ directory:

		doxygen fvens_doxygen.cfg

Of course, this requires you to have [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) installed. This will generate HTML documentation, which can be accessed through doc/html/index.htm.

Running
-------
The executables should be called with the path to a control file as input. Set OMP_NUM_THREADS to the number of threads you want to use.

		export OMP_NUM_THREADS=4
		./fvens_steady /path/to/testcases/2dcylinder/implicit.control

Control files
-------------
Examples are present in the various test cases' directories. Note that the locations of mesh files and output files should be relative to the directory from which the executable is called.

---

Copyright (C) 2016, 2017 Aditya Kashi. See LICENSE.md for terms of redistribution with/without modification and those of linking.

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

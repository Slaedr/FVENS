FVENS
=====

This is a cell-centered finite volume solver for the two-dimensional compressible Euler and Navier-Stokes equations. Unstructured grids having both triangles and quadrangles are supported. It includes MUSCL (variable extrapolation) reconstruction using either Green-Gauss or weighted least-squares methods. WENO (weighted essentially non-oscillatory), MUSCL and linear reconstructions are availble with the Van Albada limiter for MUSCL reconstruction, and the Barth-Jespersen and Venkatakrishnan limiters for linear reconstruction. A number of numerical convective fluxes are available - local Lax-Friedrichs (Rusanov), Van Leer flux vector splitting, AUSM, HLL (Harten - Lax - Van Leer), HLLC and Roe-Pike. Modified average gradients are used for viscous fluxes. 

Currently, only steady-state problems are supported. Both explicit and implicit pseudo-time stepping are avaible. Explicit time-stepping uses the forward Euler scheme while implicit time stepping uses the backward Euler scheme; both use local time-steps.

Building
--------
The following libraries are required:
- [Boost](http://www.boost.org/), present in the default package repositories of all GNU/Linux distributions
- The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix library (preferably the development version, or version 3.3.4) - needs an environment variable called EIGEN_DIR to be set to the top-level Eigen directory
- [PETSc](http://www.mcs.anl.gov/petsc/) version 3.8 for sparse linear solvers
- Optionally, [BLASTed](https://github.com/Slaedr/BLASTed) sparse linear algebra library - needs an environment variable called BLASTED_DIR to be set to the top level BLASTed directory.

OpenMP will be used if available (default builds of GCC on most GNU/Linux distributions have this, for instance).

To build, issue

		mkdir build && cd build
		cmake /path/to/src -DCMAKE_BUILD_TYPE=Debug

and for a release build with SSE 4.2 vectorization

		cmake /path/to/src -DCMAKE_BUILD_TYPE=Release -DSSE=1

`-DNOOMP=1` should be appended if OpenMP is not available. `-DSSE=1` can be replaced by `-DAVX=1`, if your CPU supports it. See the header of the top-level CMakeLists.txt for a list of all build options.  Finally,

		make -j<N>

where '\<N\>' should be replaced by the number of threads to use for building. The build is known to work with recent versions of GCC C++ (5.4 and above) and Intel C++ (2017) compilers on GNU/Linux systems.

To run the tests, run `make test` or `ctest` in the `build` directory.

To build the Doxygen documentation, please type the following command in the doc/ directory:

		doxygen fvens_doxygen.cfg

Of course, this requires you to have [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) installed. This will generate HTML documentation, which can be accessed through doc/html/index.htm.

Browsing the code
-----------------
A tags file is built when the code is built (if ctags is available), which makes navigation in Vim convenient. Using tags in Vim is [easy](http://vim.wikia.com/wiki/Browsing_programs_with_tags).

Running
-------
The executables should be called with the paths to a control file (required) and a PETSc options file as input. Set OMP_NUM_THREADS to the number of threads you want to use.

		export OMP_NUM_THREADS=4
		./fvens_steady /path/to/testcases/2dcylinder/implicit.control -options_file /path/to/testcases/2dcylinder/opts.petscrc

Make sure to set the paths of input and output files appropriately in the control file (see the next section below).

Control files
-------------
Examples are present in the various test cases' directories. Note that the locations of mesh files and output files should be relative to the directory from which the executable is called. The structure of the control files is currently very rigid; it is recommended to copy one of the existing cases' control file and modify it.

PETSc options for FVENS
-----------------------
* -mesh_reorder (string argument): If mentioned, the mesh cells will be reordered in the preprocessing stage, into one of the supported [PETSc orderings](www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatOrderingType.html).
* -matrix_free_jacobian (no argument): If mentioned, matrix finite-difference system matrix will be used, but the first-order approximate Jacobian will still be stored for the preconditioner.
* -matrix_free_difference_step (float argument): The finite difference difference step length to use in case the matrix-free solver is requested; if not mentioned, this defaults to 1e-7.
* -fvens_log_file (string argument): Prefix (path + base file name) of the file into which to write timing logs (.tlog extension), and if requested, nonlinear residual histories (.conv extension). Note that this option, if specified, overrides the corresponding option in the control file.

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

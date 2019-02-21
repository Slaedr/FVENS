FVENS
=====

This is a cell-centered finite volume solver for the two-dimensional compressible Euler and Navier-Stokes equations. Unstructured grids having both triangles and quadrangles are supported. It includes gradient computation using either Green-Gauss or weighted least-squares methods. Linear, MUSCL and limited linear reconstructions are availble. The Van Albada limiter is used for MUSCL reconstruction while the WENO (weighted essentially non-oscillatory), Barth-Jespersen or Venkatakrishnan limiters can be used for limited linear reconstruction. A number of numerical inviscid fluxes are available - local Lax-Friedrichs (Rusanov), Van Leer flux vector splitting, AUSM, HLL (Harten - Lax - Van Leer), HLLC and Roe-Pike. Modified average gradients are used for viscous fluxes. For steady-state problems, both explicit and implicit pseudo-time stepping are avaible. 

'Dimension independent code' - using the same source code for 2D and 3D problems with only recompilation needed - is a goal. If you wish to contribute, please read the [guidelines](CONTRIBUTING.md).

Features
--------
- Modular, using C++ inheritence
- Almost everything implemented here is parallelized with OpenMP
- Limited support for distributed (MPI) parallelism
- Supports 2D hybrid grids containing both triangles and quadrangles
- Several different inviscid numerical flux options available
- Two different gradient computation schemes
- Several reconstruction schemes using different limiters
- A matrix-free implicit solver implementation is available; however, it uses an approximate stored Jacobian for preconditioning

Limitations
-----------
- Limited automated testing as of now
- Two spatial dimensions at present
- Not many types of boundary conditions are currently implemented - only the simplest or most common ones
- No adaptive mesh refinement
- Uses PETSc for implicit solution, so the linear solver itself is not thread-parallel as of now

Building
--------
The following libraries and programs are required:
- [Boost](http://www.boost.org/), present in the default package repositories of all GNU/Linux distributions
- The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix library (version 3.3.5 or newer) - needs a variable called `EIGEN3_ROOT` to be set to the top-level Eigen directory
- [PETSc](http://www.mcs.anl.gov/petsc/) version 3.8 or newer for sparse linear solvers; needs `PETSC_DIR` and `PETSC_ARCH` set, unless PETSc has been installed in standard system locations.
- [Gmsh](http://gmsh.info/) 3.0 or later is required for building many of the grids for test cases. It is available in most GNU/Linux distributions' official repositories. If it is not present in a standard directory and has not been added to the `PATH` variable, one can pass
`-DCMAKE_PREFIX_PATH=/path/to/top-level/gmsh-dir` in the `cmake` command line (see below).
- Optionally, the [ADOL-C](https://projects.coin-or.org/ADOL-C) automatic differentiation library can be used for computing exact derivatives of the fluxes. Unless installed in a standard directory, an environment variable `ADOLC_DIR` needs to be set, and `-DWITH_ADOLC=1` should be passed to the CMake command line.
- Optionally, [BLASTed](https://github.com/Slaedr/BLASTed) sparse linear algebra library - needs an environment variable called `BLASTED_DIR` to be set to the top level BLASTed source directory and `BLASTED_BIN_DIR` to be set to the BLASTed build directory. `-DWITH_BLASTED=1` needs to be passed to CMake.

The variables needed can either be passed as arguments to CMake during configuration (see below) or set as environment variables. OpenMP will be used if available (default builds of GCC on most GNU/Linux distributions have this, for instance).

To build go to the FVENS root directory and issue

		mkdir build && cd build
		cmake .. 

and for a release build

		cmake .. -DCMAKE_BUILD_TYPE=Release

`-DNOOMP=1` should be appended if OpenMP is not available. `-DAVX=1` etc. can be passed if your CPU supports the corresponding vector instructions. See the header of the top-level CMakeLists.txt for a list of all build options.  Finally,

		make -j<N>

where '\<N\>' should be replaced by the number of threads to use for building. The build is known to work with recent versions of GCC C++ (5.4 and above) and Intel C++ (2017 and 2018) compilers on GNU/Linux systems.

To run the tests, run `make test` or `ctest` in the `build` directory.

To build the Doxygen documentation, type the following command in the doc/ directory:

		doxygen fvens_doxygen.cfg

This requires you to have [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) installed. It will generate HTML documentation, which can be accessed through doc/html/index.htm.

Running the code
----------------
Please refer to the [user manual](docs/user-doc.md).

Browsing the code
-----------------
A tags file is built when the code is built (if ctags is available), which makes navigation convenient. Using tags in Vim or [Spacemacs](http://spacemacs.org/) is [easy](http://vim.wikia.com/wiki/Browsing_programs_with_tags).

Known issues
------------
Please see the [issues](https://github.com/Slaedr/FVENS/issues) page.

Contributors
------------
- Luis Hernandez
- Aditya Kashi

[![Built with Spacemacs](https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg)](http://spacemacs.org)

---

Copyright (C) 2016 - 2018, Aditya Kashi. See LICENSE.md for terms of redistribution with/without modification and those of linking.

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

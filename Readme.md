FVENS
=====

A finite volume (2nd order in space) compressible Euler solver. Currently,only explicit time-stepping workds.

Building
--------
The Eigen matrix library is required. Please set EIGEN_DIR before issuing
    cmake /path/to/src -DCMAKE_BUILD_TYPE=Debug -DOMP=1

Running
-------
The executables should be called with the path to a control file as input. Set  OMP_NUM_THREADS to the number of threads you want to use.

Control files
-------------
Examples are present in the various test cases' directories. Note that the locations of mesh files and output files should be relative to the directory from which the executable is called.


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

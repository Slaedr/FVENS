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

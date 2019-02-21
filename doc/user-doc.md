Running
-------
The executables should be called with the paths to a control file (required) and a PETSc options file as input. Set `OMP_NUM_THREADS` to the number of threads you want to use, or all available CPU threads will usually be used.

		export OMP_NUM_THREADS=4
		./fvens_steady testcases/2dcylinder/implicit.ctrl -options_file testcases/2dcylinder/opts.solverc

Make sure to set the paths of input and output files appropriately in the control file (see the next section below).

Control files
-------------
Examples are present (`.ctrl` files) in the various test cases' directories. Note that the locations of mesh files and output files should be relative to the directory from which the executable is called.

Command Line Options
--------------------
* `--mesh_file` <string> If given, this overrides the mesh file specified in the control file.

PETSc options for FVENS
-----------------------
* `-mesh_reorder` (string argument): If mentioned, the mesh cells will be reordered in the preprocessing stage, into one of the supported [PETSc orderings](http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatOrderingType.html). In addition, it can be `line` or `line_<ordering>` where <ordering> is any PETSc ordering. This currently only works correctly in single-process (one MPI rank) runs.
* `-mesh_anisotropy_threshold` (float argument): Only required if `-mesh_reorder line_*` is requested. This is the minimum local grid anisotropy above which a cell will be regarded as part of a line. Roughly relates to the local aspect ratio. 10.0 to 100.0 are likely to be good values.
* `-matrix_free_jacobian` (no argument): If mentioned, matrix-free finite-difference Jacobian will be used, but the first-order approximate Jacobian will still be stored for the preconditioner.
* `-matrix_free_difference_step` (float argument): The finite difference step length to use in case the matrix-free solver is requested; if not mentioned, this defaults to 1e-7.
* `-fvens_log_file_prefix` (string argument): Prefix (path + base file name) of the file into which to write timing logs, and if requested, nonlinear residual histories (using different suffixes). Note that this option, if specified, overrides the corresponding option in the control file.

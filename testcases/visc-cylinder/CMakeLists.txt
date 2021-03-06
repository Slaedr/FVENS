
configure_file(laminar-implicit.ctrl laminar-implicit.ctrl)
configure_file(opts.solverc opts.solverc COPYONLY)
configure_file(mgopts.solverc mgopts.solverc COPYONLY)

add_subdirectory(grids)

add_test(NAME SpatialFlow_NS_2dcylinder_LeastSquares_HLLC_FunctionalRegression
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/laminar-implicit.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  --mesh_file grids/2dcylinderhybrid2.msh  --regression_test 1
  --regression_file ${CMAKE_CURRENT_SOURCE_DIR}/regr-LeastSquares_HLLC.txt
  )

add_test(NAME MPI_SpatialFlow_Explicit_NS_2dcylinder_LeastSquares_HLLC_ResidualConv
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/laminar-expl.ctrl
  --mesh_file ../2dcylinder/grids/2dcylquad1.msh)

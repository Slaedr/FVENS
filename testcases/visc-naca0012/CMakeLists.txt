
configure_file(laminar-implicit.ctrl laminar-implicit.ctrl)
configure_file(opts.solverc opts.solverc COPYONLY)
configure_file(mgopts.solverc mgopts.solverc COPYONLY)

add_subdirectory(grids)

add_test(NAME SpatialFlow_NS_NACA0012_LeastSquares_Roe_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/laminar-implicit.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  --mesh_file grids/NACA0012_lam_hybrid_1.msh)

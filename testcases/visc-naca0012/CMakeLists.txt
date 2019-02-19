
configure_file(laminar-implicit.ctrl laminar-implicit.ctrl)
configure_file(opts.solverc opts.solverc COPYONLY)
configure_file(mgopts.solverc mgopts.solverc COPYONLY)

add_subdirectory(grids)

add_test(NAME SpatialFlow_NS_NACA0012_LeastSquares_Roe_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/laminar-implicit.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  --mesh_file grids/NACA0012_lam_hybrid_1.msh)

add_test(NAME SpatialFlow_NS_NACA0012_LS_Roe_LineOrdering_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_BINARY_DIR}/tests/e_testflow 
  ${CMAKE_SOURCE_DIR}/testcases/visc-naca0012/laminar-implicit.ctrl
  -options_file ${CMAKE_SOURCE_DIR}/testcases/visc-naca0012/opts.solverc
  --mesh_file ${CMAKE_BINARY_DIR}/testcases/visc-naca0012/grids/NACA0012_lam_hybrid_1.msh
  -mesh_reorder line)

add_test(NAME SpatialFlow_NS_NACA0012_LeastSquares_Roe_MatFree_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/laminar-implicit.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc -sub_pc_factor_levels 1
  -matrix_free_jacobian -matrix_free_difference_step 1e-6
  --mesh_file grids/NACA0012_lam_hybrid_1.msh)

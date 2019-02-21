
configure_file(transonic-implicit.ctrl transonic-implicit.ctrl)
configure_file(opts.solverc opts.solverc)

add_subdirectory(grids)

add_test(NAME SpatialFlow_Euler_NACA0012_WENO_LeastSquares_HLLC_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/transonic-sanity-test-weno.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  --mesh_file grids/naca0012luo.msh)
add_test(NAME SpatialFlow_Euler_NACA0012_MUSCL_LeastSquares_HLLC_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/transonic-sanity-test-muscl.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  --mesh_file grids/naca0012luo.msh)

add_test(NAME SpatialFlow_Euler_NACA0012_MUSCL_LS_HLLC_LineOrdering_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../../tests/e_testflow 
  ${CMAKE_CURRENT_SOURCE_DIR}/transonic-sanity-test-muscl.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
  -mesh_reorder line -mesh_anisotropy_threshold 10.0
  --mesh_file grids/naca0012luo.msh)

# add_test(NAME MPI_SpatialFlow_Euler_NACA0012_MUSCL_LeastSquares_HLLC_ResidualConvergence
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ../../tests/e_testflow 
#   ${CMAKE_CURRENT_SOURCE_DIR}/transonic-sanity-test-muscl.ctrl
#   -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc
#   --mesh_file grids/naca0012luo.msh)

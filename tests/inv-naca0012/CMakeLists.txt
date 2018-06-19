
add_test(NAME SpatialFlow_Euler_NACA0012_transonic_WENO_LeastSquares_HLLC
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../e_testflow
  ${CMAKE_CURRENT_SOURCE_DIR}/transonic-naca0012.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/transonic-naca0012.solverc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/naca0012/grids/naca0012luo.msh)

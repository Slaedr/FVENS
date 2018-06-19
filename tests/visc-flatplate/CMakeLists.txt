
add_test(NAME SpatialFlow_NavierStokes_FlatPlate_LeastSquares_Roe_Quad
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ../e_testflow_conv_clcd
  ${CMAKE_CURRENT_SOURCE_DIR}/flatplate.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flatplate.solverc
  --exact_solution_file ${CMAKE_CURRENT_SOURCE_DIR}/exact_clcd_flatplate.dat
  --number_of_meshes 3 --test_type CDSF
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/visc-flatplate/grids/flatplatestructstretched)

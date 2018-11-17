
# List of control files
set(CONTROL_FILES inv-cyl-gg-hllc_tri.ctrl
  inv-cyl-ls-hllc.ctrl
  expl-cyl-ls-hllc.ctrl)

# Process them to include CMake variables
foreach(file ${CONTROL_FILES})
  message(STATUS "Configuring control file ${file}")
  configure_file(${file} ${file})
endforeach(file)

add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Tri_EntropyConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-cyl-ls-hllc.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)

add_test(NAME SpatialFlow_Euler_Cylinder_GreenGauss_HLLC_Tri_EntropyConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-cyl-gg-hllc_tri.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)

add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Quad_EntropyConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-cyl-ls-hllc.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ../../testcases/2dcylinder/grids/2dcylquad)

add_test(NAME Flow_Euler_Cylinder_Explicit_LeastSquares_HLLC_Quad_ResidualConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND sh ../flow-general/test_res_hist.sh
  ${CMAKE_CURRENT_BINARY_DIR}/expl-cyl-ls-hllc.ctrl
  --mesh_file ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad1.msh
  --log_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/2dcyl )

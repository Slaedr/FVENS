# Test executables

add_executable(exec_testdiffusion heat_steady.cpp)
target_link_libraries(exec_testdiffusion fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
  target_link_libraries(exec_testdiffusion ${BLASTED_LIB})
endif()

configure_file(implls_quad.control implls_quad.control)
configure_file(implls_tri.control implls_tri.control)

# The heat tests must still take place in the source directory as it has a hard-coded
#  command-line parsing. For now that's okay because there are no ouput files.
add_test(NAME SpatialDiffusion_LeastSquares_Quad_SolnConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  ${CMAKE_CURRENT_BINARY_DIR}/implls_quad.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc)

add_test(NAME SpatialDiffusion_LeastSquares_Tri_SolnConvergence 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  ${CMAKE_CURRENT_BINARY_DIR}/implls_tri.control -options_file ${CMAKE_CURRENT_SOURCE_DIR}/opts.solverc)


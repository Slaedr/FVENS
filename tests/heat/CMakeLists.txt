# Test executables

add_executable(exec_testdiffusion heat_steady.cpp)
target_link_libraries(exec_testdiffusion fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(exec_testdiffusion ${BLASTED_LIB})
endif()

# The heat tests must still take place in the source directory as it has a hard-coded
#  command-line parsing. For now that's okay because there are no ouput files.
add_test(NAME SpatialDiffusion_LeastSquares_Quad_SolnConvergence
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  implls_quad.control -options_file opts.solverc)
add_test(NAME SpatialDiffusion_LeastSquares_Tri_SolnConvergence 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  implls_tri.control -options_file opts.solverc)


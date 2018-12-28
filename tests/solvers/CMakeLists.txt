# Test executables
	
if(WITH_BLASTED)
  add_executable(check_bench_output testbench.cpp)
  configure_file(testbench.sh testbench.sh)

  # Tests
  add_test(NAME Benchmark_Euler_Blasted_run
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND sh testbench.sh
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark.ctrl 
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.solverc
	--mesh_file ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad2.msh )
endif()

add_executable(testmatfree testmatrixfree.cpp)
target_link_libraries(testmatfree fvens_base ${PETSC_LIB})

add_executable(testtracecomm testtracevector.cpp)
target_link_libraries(testtracecomm fvens_base)

add_test(NAME MPI_TraceVector_Comm_Rank
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${MPIEXEC} -n 2 testtracecomm
  ${CMAKE_SOURCE_DIR}/tests/heat/grids/squareunsquad1.msh
  #${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad1.msh
  )

add_test(NAME Flow_Euler_Cylinder_HLLC_MatFreeVsMat
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} testmatfree
  ${CMAKE_CURRENT_SOURCE_DIR}/matfree.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/matfree.solverc
  --mesh_file ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad3.msh)

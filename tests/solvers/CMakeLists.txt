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
	--mesh_file ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad2.msh
	-blasted_compute_preconditioner_info)
endif()

add_executable(testmatfree testmatrixfree.cpp)
target_link_libraries(testmatfree fvens_base ${PETSC_LIB})

add_executable(testtracecomm testtracevector.cpp)
target_link_libraries(testtracecomm fvens_base)

add_test(NAME MPI_TraceVector_Comm_1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${MPIEXEC} -n 3 testtracecomm
  ${CMAKE_SOURCE_DIR}/tests/heat/grids/squareunsquad3.msh
  )

add_test(NAME MPI_TraceVector_Comm_2
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 testtracecomm
  ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad2.msh
  )

add_test(NAME Flow_Euler_Cylinder_HLLC_MatFreeVsMat
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} testmatfree
  ${CMAKE_CURRENT_SOURCE_DIR}/matfree.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/matfree.solverc
  --mesh_file ${CMAKE_BINARY_DIR}/testcases/2dcylinder/grids/2dcylquad3.msh)

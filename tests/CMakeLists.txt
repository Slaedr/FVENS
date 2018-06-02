# Test executables
	
if(SLURM)
	set(SEQEXEC "srun")
	set(SEQTASKS "-n 1")
	set(MPIEXEC "srun")
else(SLURM)
	set(SEQEXEC "")
	set(SEQTASKS "")
	set(MPIEXEC "mpirun")
endif(SLURM)

# Gmsh for generating meshes
find_program(GMSH_EXEC gmsh)
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Gmsh not found")
else()
  message(STATUS "Found Gmsh at ${GMSH_EXEC}")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(exec_testmesh mesh.cpp)
target_link_libraries(exec_testmesh fvens_base)

add_library(testflowspatial testflowspatial.cpp)
target_link_libraries(testflowspatial fvens_base)

add_executable(e_testflow_spatial flowspatial.cpp)
target_link_libraries(e_testflow_spatial testflowspatial)

add_executable(e_testflow_pseudotime flowpseudotime.cpp)
target_link_libraries(e_testflow_pseudotime fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_pseudotime ${BLASTED_LIB})
endif()

add_executable(exec_testdiffusion heat_steady.cpp)
target_link_libraries(exec_testdiffusion fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(exec_testdiffusion ${BLASTED_LIB})
endif()

add_executable(e_testflow flow_solve.cpp)
target_link_libraries(e_testflow fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow ${BLASTED_LIB})
endif()

add_executable(e_testflow_conv flow_conv.cpp)
target_link_libraries(e_testflow_conv fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_conv ${BLASTED_LIB})
endif()

add_executable(e_testflow_conv_clcd flow_clcd_conv.cpp)
target_link_libraries(e_testflow_conv_clcd fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_conv_clcd ${BLASTED_LIB})
endif()

# Tests

add_test(NAME Mesh_Topology_ElemSurrElem
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  esup ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcylinderhybrid.msh)
add_test(NAME Mesh_Periodic
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh periodic
  ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)

add_test(NAME MeshUtils_LevelSchedule WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  levelschedule ${CMAKE_CURRENT_SOURCE_DIR}/input/squarecoarse.msh 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/squarecoarselevels.dat)
add_test(NAME MeshUtils_LevelSchedule_Internal WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  levelscheduleInternal ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcylinderhybrid.msh)

add_test(NAME SpatialFlow_BC_Walls WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg wall_boundaries
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)

add_test(NAME SpatialFlow_Walltest_HLLC WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux HLLC
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_Walltest_Roe WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux ROE
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_Walltest_AUSM WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND  ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux AUSM
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_Walltest_AUSMPlus WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND  ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux AUSMPLUS
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_Walltest_HLL WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux HLL
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_Walltest_LLF WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_spatial
  ${CMAKE_CURRENT_SOURCE_DIR}/input/test.cfg numerical_flux LLF
  --mesh_file ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)

add_test(NAME PseudotimeFlow_exception_nanorinf WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_pseudotime
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/testexception.control -test_type exception_nanorinf
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv_cyl.petscrc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/visc-naca0012/grids/NACA0012_lam_hybrid_1.msh)

# The heat tests must still take place in the source directory as it has a hard-coded
#  command-line parsing. For now that's okay because there are no ouput files.
add_test(NAME SpatialDiffusion_LeastSquares_Quad WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  heat/implls_quad.control -options_file heat/opts.petscrc)
add_test(NAME SpatialDiffusion_LeastSquares_Tri WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testdiffusion
  heat/implls_tri.control -options_file heat/opts.petscrc)

add_subdirectory(isentropic_vortex)

add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv-cyl-ls-hllc_tri.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv_cyl.petscrc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)
add_test(NAME SpatialFlow_Euler_Cylinder_GreenGauss_HLLC_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv-cyl-gg-hllc_tri.control -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv_cyl.petscrc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)
add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Quad
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv-cyl-ls-hllc_quad.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/inv_cyl.petscrc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylquad)

add_test(NAME SpatialFlow_Euler_NACA0012_transonic_WENO_LeastSquares_HLLC
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/transonic-naca0012.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/transonic-naca0012.petscrc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/naca0012/grids/naca0012luo.msh)

add_test(NAME SpatialFlow_NavierStokes_FlatPlate_LeastSquares_Roe_Quad
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv_clcd
  ${CMAKE_CURRENT_SOURCE_DIR}/flow/flatplate.control
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/flatplate.petscrc
  -exact_solution_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/exact_clcd_flatplate.dat
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/visc-flatplate/grids/flatplatestructstretched)

if(WITH_BLASTED)
  add_test(NAME Benchmark_Euler_Blasted_run
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	  #COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/benchtest.sh
	  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_BINARY_DIR}/bench_threads_async
	  ${CMAKE_CURRENT_SOURCE_DIR}/flow/small-2dcylinder.control 
	  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow/benchmark.petscrc
	  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylquad2.msh )
endif()

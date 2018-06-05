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

add_executable(e_testflow_pseudotime flowpseudotime.cpp)
target_link_libraries(e_testflow_pseudotime fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_pseudotime ${BLASTED_LIB})
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

# List of control files
set(CONTROL_FILES inv-2dcyl/inv-cyl-gg-hllc_tri.ctrl
  inv-2dcyl/inv-cyl-ls-hllc_tri.ctrl
  inv-2dcyl/inv-cyl-ls-hllc_quad.ctrl)
# Process them to include CMake variables
foreach(file ${CONTROL_FILES})
  message(STATUS "Configuring control file ${file}")
  configure_file(${file} ${file})
endforeach(file)

# Tests

add_subdirectory(utils)

add_subdirectory(mesh)

add_subdirectory(heat)

add_subdirectory(flow-general)

add_test(NAME PseudotimeFlow_exception_nanorinf WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_pseudotime
  ${CMAKE_CURRENT_SOURCE_DIR}/flow-general/testexception.ctrl
  --test_type exception_nanorinf
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flow-general/testexception.solverc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/visc-naca0012/grids/NACA0012_lam_hybrid_1.msh)

#add_subdirectory(isentropic-vortex)

add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-2dcyl/inv-cyl-ls-hllc_tri.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv-2dcyl/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)
add_test(NAME SpatialFlow_Euler_Cylinder_GreenGauss_HLLC_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-2dcyl/inv-cyl-gg-hllc_tri.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv-2dcyl/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylinder)
add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Quad
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv
  ${CMAKE_CURRENT_BINARY_DIR}/inv-2dcyl/inv-cyl-ls-hllc_quad.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv-2dcyl/inv_cyl.solverc
  --number_of_meshes 4
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/2dcylinder/grids/2dcylquad)

add_test(NAME SpatialFlow_Euler_NACA0012_transonic_WENO_LeastSquares_HLLC
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow
  ${CMAKE_CURRENT_SOURCE_DIR}/inv-naca0012/transonic-naca0012.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/inv-naca0012/transonic-naca0012.solverc
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/naca0012/grids/naca0012luo.msh)

add_test(NAME SpatialFlow_NavierStokes_FlatPlate_LeastSquares_Roe_Quad
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testflow_conv_clcd
  ${CMAKE_CURRENT_SOURCE_DIR}/visc-flatplate/flatplate.ctrl
  -options_file ${CMAKE_CURRENT_SOURCE_DIR}/visc-flatplate/flatplate.solverc
  --exact_solution_file ${CMAKE_CURRENT_SOURCE_DIR}/visc-flatplate/exact_clcd_flatplate.dat
  --number_of_meshes 3 --test_type CDSF
  --mesh_file ${CMAKE_SOURCE_DIR}/testcases/visc-flatplate/grids/flatplatestructstretched)


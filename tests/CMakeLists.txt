# Test executables

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(exec_testmesh mesh.cpp)
target_link_libraries(exec_testmesh fvens_base)

add_library(testflowspatial testflowspatial.cpp)
target_link_libraries(testflowspatial fvens_base)

add_executable(e_testflow_convspatial flowfv.cpp)
target_link_libraries(e_testflow_convspatial testflowspatial)

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

add_test(NAME Mesh_Topology_ElemSurrElem COMMAND exec_testmesh esup ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcylinderhybrid.msh)
add_test(NAME Mesh_Periodic COMMAND exec_testmesh periodic ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME MeshUtils_LevelSchedule WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND exec_testmesh levelschedule input/squarecoarse.msh input/squarecoarselevels.dat)
add_test(NAME MeshUtils_LevelSchedule_Internal WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND exec_testmesh levelscheduleInternal input/2dcylinderhybrid.msh)

add_test(NAME SpatialFlow_BC_Walls WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_convspatial input/test.cfg wall_boundaries)

add_test(NAME SpatialFlow_Walltest_HLLC WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_convspatial input/test.cfg numerical_flux HLLC)
add_test(NAME SpatialFlow_Walltest_Roe WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_convspatial input/test.cfg numerical_flux ROE)
add_test(NAME SpatialFlow_Walltest_AUSM WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND  e_testflow_convspatial input/test.cfg numerical_flux AUSM)
add_test(NAME SpatialFlow_Walltest_AUSMPlus WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND  e_testflow_convspatial input/test.cfg numerical_flux AUSMPLUS)
add_test(NAME SpatialFlow_Walltest_HLL WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND  e_testflow_convspatial input/test.cfg numerical_flux HLL)
add_test(NAME SpatialFlow_Walltest_LLF WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_convspatial input/test.cfg numerical_flux LLF)

add_test(NAME SpatialDiffusion_LeastSquares_Quad WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND exec_testdiffusion heat/implls_quad.control -options_file heat/opts.petscrc)
add_test(NAME SpatialDiffusion_LeastSquares_Tri WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND exec_testdiffusion heat/implls_tri.control -options_file heat/opts.petscrc)

add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Tri WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_conv flow/inv-cyl-ls-hllc_tri.control -options_file flow/inv_cyl.petscrc)
add_test(NAME SpatialFlow_Euler_Cylinder_GreenGauss_HLLC_Tri WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_conv flow/inv-cyl-gg-hllc_tri.control -options_file flow/inv_cyl.petscrc)
add_test(NAME SpatialFlow_Euler_Cylinder_LeastSquares_HLLC_Quad WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_conv flow/inv-cyl-ls-hllc_quad.control -options_file flow/inv_cyl.petscrc)

add_test(NAME SpatialFlow_Euler_NACA0012_transonic_WENO_LeastSquares_HLLC WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMAND e_testflow tests/flow/transonic-naca0012.control -options_file tests/flow/transonic-naca0012.petscrc)

add_test(NAME SpatialFlow_NavierStokes_FlatPlate_LeastSquares_Roe_Quad WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND e_testflow_conv_clcd flow/flatplate.control -options_file flow/flatplate.petscrc)

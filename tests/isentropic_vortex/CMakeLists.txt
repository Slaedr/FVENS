# Test executables

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(test_isentropicvortex isentropicvortex_main.cpp isentropicvortex.cpp)
target_link_libraries(test_isentropicvortex fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
  target_link_libraries(test_isentropicvortex ${BLASTED_LIB})
endif(WITH_BLASTED)

set(isentropicvortex_meshfiles
  ${CMAKE_CURRENT_BINARY_DIR}/grids/dom0.msh 
  ${CMAKE_CURRENT_BINARY_DIR}/grids/dom1.msh 
  ${CMAKE_CURRENT_BINARY_DIR}/grids/dom2.msh 
  #${CMAKE_CURRENT_BINARY_DIR}/grids/dom3.msh 
  )
foreach(imesh 0 1 2)
  math(EXPR imeshp1 ${imesh}+1)
  add_custom_command(
	COMMENT "Build grid needed for isentropic vortex test"
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/grids/dom${imesh}.msh
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/grids
	COMMAND gmsh -setnumber refine ${imeshp1} -2 -o ${CMAKE_CURRENT_BINARY_DIR}/grids/dom${imesh}.msh
	-v 2 grids/dom.geo
	MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/grids/dom.geo"
	)
endforeach(imesh)
add_custom_target(isentropicvortex_meshes DEPENDS ${isentropicvortex_meshfiles})
add_dependencies(test_isentropicvortex isentropicvortex_meshes)

# Tests

add_test(NAME SpatialTemportalFlow_Euler_Vortex_TVDRK_LeastSquares_Roe_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND test_isentropicvortex ${CMAKE_CURRENT_SOURCE_DIR}/isentropicvortex.control
  ${CMAKE_CURRENT_SOURCE_DIR}/vortex.params -number_of_meshes 3
  )


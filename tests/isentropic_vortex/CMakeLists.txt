# Test executables

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(test_isentropicvortex isentropicvortex_main.cpp isentropicvortex.cpp)
target_link_libraries(test_isentropicvortex fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(test_isentropicvortex ${BLASTED_LIB})
endif()

# Tests

add_test(NAME SpatialTemportal_Euler_Vortex_TVDRK_LeastSquares_Roe_Tri
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND test_isentropicvortex isentropicvortex.control vortex.params)


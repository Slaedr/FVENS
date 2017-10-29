# Test executables

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(testmesh mesh.cpp)
target_link_libraries(testmesh fvens_base)

add_executable(testflowfv flowfv.cpp)
target_link_libraries(testflowfv fvens_base)

# Tests

enable_testing()

add_test(NAME Mesh COMMAND testmesh ${CMAKE_CURRENT_SOURCE_DIR}/input/testperiodic.msh)
add_test(NAME SpatialFlow_BC_NoslipWall WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND testflowfv input/test.control no_slip_boundaries)

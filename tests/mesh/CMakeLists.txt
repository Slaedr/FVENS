# Test executables
	
add_executable(exec_testmesh mesh.cpp)
target_link_libraries(exec_testmesh fvens_base)

# Tests

add_test(NAME Mesh_Topology_ElemSurrElem
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  esup ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/2dcylinderhybrid.msh)
add_test(NAME Mesh_Periodic
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh periodic
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testperiodic.msh)

add_test(NAME MeshUtils_LevelSchedule WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  levelschedule ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/squarecoarse.msh 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/squarecoarselevels.dat)
add_test(NAME MeshUtils_LevelSchedule_Internal WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  levelscheduleInternal ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/2dcylinderhybrid.msh)


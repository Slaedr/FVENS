# Test executables
	
add_executable(exec_testmesh mesh.cpp)
target_link_libraries(exec_testmesh fvens_base)

add_executable(exec_disttestmesh distributedmesh.cpp)
target_link_libraries(exec_disttestmesh fvens_base)

add_executable(exec_testlineordering testlineordering.cpp)
target_link_libraries(exec_testlineordering fvens_base)

add_executable(exec_testhybridlineordering testhybridlineordering.cpp)
target_link_libraries(exec_testhybridlineordering fvens_base)

# Tests

add_test(NAME Mesh_Topology_ElemSurrElem
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  esup ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/2dcylinderhybrid.msh)
add_test(NAME Mesh_Topology_FaceStructure
  COMMAND ${SEQEXEC} ${SEQTASKS} exec_testmesh
  intfac ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/2dcylinderhybrid.msh)
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

add_test(NAME MeshUtils_findLines_smallmesh
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testlineordering
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testanisotropic.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testanisotropic-checkdata.txt
  10.0
  )

add_test(NAME MeshUtils_hybridLineOrdering_natural_smallmesh
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/exec_testhybridlineordering
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testanisotropic.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testanisotropic-checkdata.txt
  natural
  10.0
  )

add_test(NAME MeshPartition_Sanity WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${MPIEXEC} -n 1 ${CMAKE_CURRENT_BINARY_DIR}/exec_disttestmesh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid-distb.dat
  )

add_test(NAME MeshPartition_SubdomainRestriction WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/exec_disttestmesh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid_part1.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid_part2.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid_part3.msh
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid-distb_part1.dat
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid-distb_part2.dat
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testhybrid-distb_part3.dat
  )

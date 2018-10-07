# Generate meshes
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Testcases: visc-2dflatplate meshes not built because Gmsh was not found")
else()

  set(visc-2dflatplate_meshfiles
    ${CMAKE_CURRENT_BINARY_DIR}/flatplatestruct2.msh 
	)

  add_custom_command(
    COMMENT "Building grid needed for testcase visc-2dflatplate"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/flatplatestruct2.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/flatplatestruct2.msh
      -v 2
      flatplatestruct.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/flatplatestruct.geo"
    )

  add_custom_target(visc-2dflatplate_meshes DEPENDS ${visc-2dflatplate_meshfiles})
  add_dependencies(fvens_steady visc-2dflatplate_meshes)
endif()


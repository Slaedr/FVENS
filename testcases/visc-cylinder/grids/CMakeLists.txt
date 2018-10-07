# Generate meshes
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Testcases: visc-2dcylinder meshes not built because Gmsh was not found")
else()

  set(visc-2dcylinder_meshfiles
    ${CMAKE_CURRENT_BINARY_DIR}/2dcylinderhybrid2.msh 
	${CMAKE_CURRENT_BINARY_DIR}/2dcylinderhybrid4.msh 
	)

  foreach(imesh 2 4)
	add_custom_command(
	  COMMENT "Building grid needed for testcase visc-2dcylinder"
	  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylinderhybrid${imesh}.msh
	  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	  COMMAND ${GMSH_EXEC} -setnumber refine ${imesh} -2 -format msh2
	    -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylinderhybrid${imesh}.msh
	    -v 2
	    2dcylinderhybrid.geo
	  MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylinderhybrid.geo"
	  )
  endforeach(imesh)

  add_custom_target(visc-2dcylinder_meshes DEPENDS ${visc-2dcylinder_meshfiles})
  add_dependencies(fvens_steady visc-2dcylinder_meshes)
endif()

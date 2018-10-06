
set(MESHBASE "2dcylinder")
foreach(imesh 0 1 2 3)
	configure_file(${MESHBASE}${imesh}.msh ${MESHBASE}${imesh}.msh COPYONLY)
endforeach(imesh)

# Copy Gmsh source file in case user wants to generate more meshes
configure_file(2dcylquad.geo 2dcylquad.geo COPYONLY)

# Generate meshes
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Testcases: 2dcylinder meshes not built because Gmsh was not found")
else()

  set(2dcylinder_meshfiles
  	${CMAKE_CURRENT_BINARY_DIR}/2dcylquad0.msh
  	${CMAKE_CURRENT_BINARY_DIR}/2dcylquad1.msh
	${CMAKE_CURRENT_BINARY_DIR}/2dcylquad2.msh 
	${CMAKE_CURRENT_BINARY_DIR}/2dcylquad3.msh 
	${CMAKE_CURRENT_BINARY_DIR}/2dcylquad4.msh 
	)

#  foreach(imeshh 2 3 4)
#    message(STATUS "imeshh = ${imeshh}")
#	math(EXPR refval "2 ^ ${imeshh}+1")
#	message(STATUS "refval = ${refval}")
#	add_custom_command(
#	  COMMENT "Building grid needed for testcase 2dcylinder"
#	  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad${imeshh}.msh
#	  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#	  COMMAND ${GMSH_EXEC} -setnumber refine ${refval} -2 -format msh2
#	    -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad${imeshh}.msh
#	    -v 2
#	    2dcylquad.geo
#	  MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
#	  )
#  endforeach(imeshh)

  add_custom_command(
    COMMENT "Building grid needed for testcase 2dcylinder"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad0.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -setnumber refine 1 -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad0.msh
      -v 2
      2dcylquad.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
    )
  add_custom_command(
    COMMENT "Building grid needed for testcase 2dcylinder"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad1.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -setnumber refine 2 -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad1.msh
      -v 2
      2dcylquad.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
    )
  add_custom_command(
    COMMENT "Building grid needed for testcase 2dcylinder"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad2.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -setnumber refine 4 -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad2.msh
      -v 2
      2dcylquad.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
    )
  add_custom_command(
    COMMENT "Building grid needed for testcase 2dcylinder"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad3.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -setnumber refine 8 -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad3.msh
      -v 2
      2dcylquad.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
    )
  add_custom_command(
    COMMENT "Building grid needed for testcase 2dcylinder"
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad4.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GMSH_EXEC} -setnumber refine 16 -2 -format msh2
      -o ${CMAKE_CURRENT_BINARY_DIR}/2dcylquad4.msh
      -v 2
      2dcylquad.geo
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/2dcylquad.geo"
    )

  add_custom_target(2dcylinder_meshes DEPENDS ${2dcylinder_meshfiles})
  add_dependencies(fvens_steady 2dcylinder_meshes)
endif()

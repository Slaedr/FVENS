
# List of control files
set(CONTROL_FILES gg-hllc_tri.ctrl
  ls-hllc_tri.ctrl)
# Process them to include CMake variables
foreach(file ${CONTROL_FILES})
  message(STATUS "Configuring control file ${file}")
  configure_file(${file} ${file})
endforeach(file)

# Generate meshes
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Gaussian bump test not built because Gmsh was not found")
else()

  set(gaussianbump_meshfiles
	${CMAKE_CURRENT_BINARY_DIR}/grids/channel0.msh 
	${CMAKE_CURRENT_BINARY_DIR}/grids/channel1.msh 
	${CMAKE_CURRENT_BINARY_DIR}/grids/channel2.msh 
	${CMAKE_CURRENT_BINARY_DIR}/grids/channel3.msh 
	${CMAKE_CURRENT_BINARY_DIR}/grids/channel4.msh 
	)

  foreach(imesh 0 1 2 3 4)
	math(EXPR imeshp1 ${imesh}+1)
	add_custom_command(
	  COMMENT "Build grid needed for Gaussian bump test"
	  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/grids/channel${imesh}.msh
	  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	  COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/grids
	  COMMAND ${GMSH_EXEC} -setnumber ref ${imeshp1} -2 -format msh2
	    -o ${CMAKE_CURRENT_BINARY_DIR}/grids/channel${imesh}.msh
	    -v 2
	    gaussian_channel.geo
	  MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/gaussian_channel.geo"
	  )
  endforeach(imesh)

  add_custom_target(gaussianbump_meshes DEPENDS ${gaussianbump_meshfiles})
  add_dependencies(e_testflow_conv gaussianbump_meshes)

  add_test(NAME SpatialFlow_Euler_GaussianBump_LeastSquares_WENO_HLLC_Tri_EntropyConvergence
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${SEQEXEC} ${SEQTASKS} ../e_testflow_conv
	${CMAKE_CURRENT_BINARY_DIR}/ls-hllc_tri.ctrl
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/ibump.solverc
	--number_of_meshes 4
	--mesh_file grids/channel)
  
  add_test(NAME SpatialFlow_Euler_GaussianBump_GreenGauss_HLLC_Tri_EntropyConvergence
  	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  	COMMAND ${SEQEXEC} ${SEQTASKS} ../e_testflow_conv
  	${CMAKE_CURRENT_BINARY_DIR}/gg-hllc_tri.ctrl
  	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/ibump.solverc
  	--number_of_meshes 4
  	--mesh_file grids/channel)

endif()


# Generate meshes

set(FLATPLATE_GRIDS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/testcases/visc-flatplate/grids")
set(FLATPLATE_GRIDS_DIR "${CMAKE_BINARY_DIR}/testcases/visc-flatplate/grids")

configure_file(${FLATPLATE_GRIDS_SOURCE_DIR}/flatplatestructstretched.geo 
	${FLATPLATE_GRIDS_DIR}/flatplatestructstretched.geo COPYONLY)

if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Gaussian bump test not built because Gmsh was not found")
else()

  set(flatplate_meshfiles
	${FLATPLATE_GRIDS_DIR}/flatplatestructstretched0.msh 
	${FLATPLATE_GRIDS_DIR}/flatplatestructstretched1.msh 
	${FLATPLATE_GRIDS_DIR}/flatplatestructstretched2.msh 
	#${FLATPLATE_GRIDS_DIR}/grids/flatplatestructstretched3.msh 
	)

  add_custom_command(
    COMMENT "Build grid needed for flatplate CD test"
	#OUTPUT ${FLATPLATE_GRIDS_DIR}/flatplatestructstretched2.msh
	OUTPUT ${flatplate_meshfiles}
    WORKING_DIRECTORY ${FLATPLATE_GRIDS_DIR}
    COMMAND ${GMSH_EXEC} - -v 0 ${FLATPLATE_GRIDS_DIR}/flatplatestructstretched.geo
    MAIN_DEPENDENCY "${FLATPLATE_GRIDS_DIR}/flatplatestructstretched.geo"
    )

  add_custom_target(flatplate_meshes DEPENDS ${flatplate_meshfiles})
  add_dependencies(e_testflow_conv_clcd flatplate_meshes)
  
  add_test(NAME SpatialFlow_NS_FlatPlate_LeastSquares_Roe_Struct_CDConvergence
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ../e_testflow_conv_clcd
    ${CMAKE_CURRENT_SOURCE_DIR}/flatplate.ctrl
    -options_file ${CMAKE_CURRENT_SOURCE_DIR}/flatplate.solverc
    --exact_solution_file ${CMAKE_CURRENT_SOURCE_DIR}/exact_clcd_flatplate.dat
    --number_of_meshes 3 --test_type CDSF
	--mesh_file ${FLATPLATE_GRIDS_DIR}/flatplatestructstretched)

endif()

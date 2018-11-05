# Test executables
	
# Gmsh for generating meshes
find_program(GMSH_EXEC gmsh)
if(${GMSH_EXEC} STREQUAL "GMSH_EXEC-NOTFOUND")
  message(WARNING "Gmsh not found")
else()
  message(STATUS "Found Gmsh at ${GMSH_EXEC}")
endif()

add_executable(e_testflow_pseudotime flowpseudotime.cpp)
target_link_libraries(e_testflow_pseudotime fvens_base ${PETSC_LIB} ${ADOLC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_pseudotime ${BLASTED_LIB})
endif()

add_executable(e_testflow flow_solve.cpp)
target_link_libraries(e_testflow fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow ${BLASTED_LIB})
endif()

add_executable(e_testflow_conv flow_conv.cpp)
target_link_libraries(e_testflow_conv fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_conv ${BLASTED_LIB})
endif()

add_executable(e_testflow_conv_clcd flow_clcd_conv.cpp)
target_link_libraries(e_testflow_conv_clcd fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(e_testflow_conv_clcd ${BLASTED_LIB})
endif()

# Tests

add_subdirectory(utils)

add_subdirectory(mesh)

add_subdirectory(finite-volume)

add_subdirectory(heat)

add_subdirectory(flow-general)

#add_subdirectory(isentropic-vortex)

add_subdirectory(inv-2dcyl)
add_subdirectory(inv-gaussianbump)

add_subdirectory(visc-flatplate)


add_subdirectory(2dcylinder)
add_subdirectory(naca0012)
add_subdirectory(visc-cylinder)
add_subdirectory(visc-flatplate)
add_subdirectory(visc-naca0012)

configure_file(defaults.solverc ${CMAKE_BINARY_DIR}/defaults.solverc COPYONLY)

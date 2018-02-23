set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# libraries to be compiled

add_library(fvens_base aodesolver.cpp alinalg.cpp aspatial.cpp afactory.cpp 
	areconstruction.cpp agradientschemes.cpp anumericalflux.cpp aoutput.cpp amesh2dh.cpp aphysics.cpp
	autilities.cpp aarray2d.cpp)
target_link_libraries(fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(fvens_base ${BLASTED_LIB})
endif()

# for the final executable(s)

add_executable(fvens_steady fvens_steady.cpp)
target_link_libraries(fvens_steady fvens_base)

add_subdirectory(utilities)



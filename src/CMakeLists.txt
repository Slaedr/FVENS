set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# libraries to be compiled

add_library(fvens_base utilities/aoptionparser.cpp utilities/afactory.cpp utilities/casesolvers.cpp
	utilities/aerrorhandling.cpp
	ode/aodesolver.cpp linalg/alinalg.cpp 
	spatial/aspatial.cpp spatial/areconstruction.cpp spatial/agradientschemes.cpp 
	spatial/anumericalflux.cpp physics/aphysics.cpp spatial/aoutput.cpp 
	mesh/ameshutils.cpp mesh/amesh2dh.cpp utilities/aarray2d.cpp)
target_link_libraries(fvens_base ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(fvens_base ${BLASTED_LIB})
endif()

# for the final executable(s)

add_executable(fvens_steady fvens_steady.cpp)
target_link_libraries(fvens_steady fvens_base)

add_subdirectory(utilities)



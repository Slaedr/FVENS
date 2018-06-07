set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})


# libraries to be compiled

add_library(fvens_parsing_errh utilities/aoptionparser.cpp
  utilities/controlparser.cpp
  utilities/aerrorhandling.cpp)
target_link_libraries(fvens_parsing_errh ${PETSC_LIB} ${Boost_LIBRARIES})
if(CXX_COMPILER_INTEL)
  target_compile_options(fvens_parsing_errh PRIVATE "-Wno-return-type")
endif()

add_library(fvens_base utilities/afactory.cpp utilities/casesolvers.cpp
	ode/aodesolver.cpp linalg/alinalg.cpp 
	spatial/aspatial.cpp spatial/areconstruction.cpp spatial/agradientschemes.cpp 
	spatial/anumericalflux.cpp physics/aphysics.cpp spatial/aoutput.cpp 
	mesh/ameshutils.cpp mesh/amesh2dh.cpp utilities/aarray2d.cpp)
target_link_libraries(fvens_base fvens_parsing_errh ${PETSC_LIB})
if(WITH_BLASTED)
	target_link_libraries(fvens_base ${BLASTED_LIB})
  endif()
target_link_libraries(fvens_base ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

# for the final executable(s)

add_executable(fvens_steady fvens_steady.cpp)
target_link_libraries(fvens_steady fvens_base)

add_subdirectory(utilities)



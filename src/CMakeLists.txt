cmake_minimum_required(VERSION 2.8)

# The environment variable EIGEN_DIR needs to be set to the path where
# the Eigen source directory is present.

project (fvens)

# to compile with OpenMP
if(OMP)
	message(STATUS "Compiling with OpenMP")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ftree-vectorizer-verbose=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -qopt-report=2")
endif()

# Eigen
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
include_directories($ENV{EIGEN_DIR})

# PETSc
if(WITH_PETSC)
	find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
	include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
	message(STATUS "Building with PETSc found at ${PETSC_LIB}")
endif()

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -UDEBUG -DNDEBUG")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# libraries to be compiled
add_library(fvens_base aspatial.cpp areconstruction.cpp alimiter.cpp anumericalflux.cpp aoutput.cpp amesh2dh.cpp adatastructures.cpp)

add_library(fvens_time aodesolver.cpp alinalg.cpp)
target_link_libraries(fvens_time fvens_base)
#if(WITH_PETSC)
#	target_link_libraries(fvens_time ${PETSC_LIB})
#endif()

# for the final executable(s)

add_subdirectory(utilities)

add_executable(fvense explicitsolver.cpp)
target_link_libraries(fvense fvens_time)

add_executable(steadyfvensi steadyimplicitsolver.cpp)
target_link_libraries(steadyfvensi fvens_time)

#add_executable(steadylufvensi steadyLUimplicitsolver.cpp)
#target_link_libraries(steadylufvensi fvens_implicit)

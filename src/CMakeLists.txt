cmake_minimum_required(VERSION 2.8)

# The environment variable EIGEN_DIR needs to be set to the path where
# the Eigen source directory is present.
# The environment variable BLASTED_DIR needs to set to the root of the BLASTed linear algebra library.
# It is assumed that the BLASTed binary is ${BLASTED_DIR}/build/libblasted.a

# Pass -DOMP=1 to compile with OpenMP
# Pass -DMICKNC=1 to compile for Xeon Phi Knights Corner.

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

if(MICKNC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	message(STATUS "Compiling for KNC")
endif()

# Eigen
include_directories($ENV{EIGEN_DIR})

# BLASTed
include_directories($ENV{BLASTED_DIR}/src)

# PETSc
if(WITH_PETSC)
	find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
	include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
	message(STATUS "Building with PETSc found at ${PETSC_LIB}")
endif()

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -UDEBUG -DNDEBUG -Wall")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# libraries to be compiled
add_library(fvens_base aodesolver.cpp alinalg.cpp aspatial.cpp areconstruction.cpp alimiter.cpp anumericalflux.cpp aoutput.cpp amesh2dh.cpp)

#if(WITH_PETSC)
#	target_link_libraries(fvens_time ${PETSC_LIB})
#endif()

# for the final executable(s)

#add_subdirectory(utilities)

add_executable(fvens_steady fvens_steady.cpp)
target_link_libraries(fvens_steady fvens_base)

#add_executable(heat_steady heat_steady.cpp)
#target_link_libraries(heat_steady fvens_base)


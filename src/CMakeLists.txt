cmake_minimum_required(VERSION 2.8)

# The environment variable EIGEN_DIR needs to be set to the path of the Eigen top-level directory.
# The environment variable BLASTED_DIR needs to set to the root of the BLASTed linear algebra library.

# Pass -DNOOMP=1 to compile without OpenMP
# Pass -DMICKNC=1 to compile for Xeon Phi Knights Corner.
# Pass -DSSE=1 to compile with SSE 4.2 instructions; ignored when compiling for KNC.
# Pass -DAVX=1 to compile with AVX instructions.
# Pass -DPROFILE=1 for profiling.

project (fvens)

set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror")
set (CMAKE_CXX_FLAGS_DEBUG " ")
set (CMAKE_CXX_FLAGS_RELEASE " ")

# to compile with OpenMP
if(NOOMP)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
	message(STATUS "Compiling without OpenMP")
else()
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		message(STATUS "Compiling with OpenMP for GNU")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
		message(STATUS "Compiling with OpenMP for Intel")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-loop-inline-vec-optimized-missed=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=2")
endif()

if(MICKNC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	message(STATUS "Compiling for KNC")
	if(NOT("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
		message(SEND_ERROR "Need to use Intel compiler for Xeon Phi!")
	endif()
else()
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
	message(STATUS "Compiling for x86_64")
	if(SSE)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2")
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with SSE4.2")
		endif()
	elseif(AVX)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX")
		endif()
	elseif(KNL)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX")
		endif()
	endif()
endif()

# profiling
if(PROFILE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG -DNDEBUG")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# libraries to be compiled
add_library(fvens_base aodesolver.cpp alinalg.cpp aspatial.cpp afactory.cpp 
	areconstruction.cpp agradientschemes.cpp anumericalflux.cpp aoutput.cpp amesh2dh.cpp aphysics.cpp
	autilities.cpp)

if(WITH_PETSC)
	target_link_libraries(fvens_base ${PETSC_LIB})
endif()

# for the final executable(s)

add_executable(fvens_steady fvens_steady.cpp)
target_link_libraries(fvens_steady fvens_base)

add_executable(heat_steady heat_steady.cpp)
target_link_libraries(heat_steady fvens_base)

# Tags

set_source_files_properties(tags PROPERTIES GENERATED true)
add_custom_target ( tags
    COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} 
  )

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
    message(STATUS "Found ctags: ${CTAGS_PATH}")
    add_dependencies (fvens_steady tags)
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)

# Tests
add_subdirectory(tests)

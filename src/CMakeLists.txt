cmake_minimum_required(VERSION 2.8)

project (fvens)

# set compiler
#set (CMAKE_C_COMPILER $ENV{GC})
#set (CMAKE_CXX_COMPILER $ENV{GPP})

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
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# libraries to be compiled
add_library(fvens_base areconstruction.cpp alimiter.cpp anumericalflux.cpp aoutput.cpp amesh2dh.cpp amatrix.cpp adatastructures.cpp)

add_library(fvens_explicit aexplicitsolver.cpp)
target_link_libraries(fvens_explicit fvens_base)

add_library(fvens_implicit aimplicitsolver.cpp alinalg.cpp)
target_link_libraries(fvens_implicit fvens_base)

# for the final executable(s)

add_subdirectory(utilities)

add_executable(fvense explicitsolver.cpp)
target_link_libraries(fvense fvens_explicit)

add_executable(steadylufvensi steadyLUimplicitsolver.cpp)
target_link_libraries(steadylufvensi fvens_implicit)

add_executable(steadyfvensi steadyimplicitsolver.cpp)
target_link_libraries(steadyfvensi fvens_implicit)

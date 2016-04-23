cmake_minimum_required(VERSION 2.8)

project (fvens)

# set compiler
#set (CMAKE_C_COMPILER $ENV{GC})
#set (CMAKE_CXX_COMPILER $ENV{GPP})

# to compile with OpenMP
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# set compile options
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb")

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -DDEBUG=1")
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

add_library(fvens areconstruction.cpp alimiter.cpp anumericalflux.cpp aexplicitsolver.cpp amesh2dh.cpp aoutput.cpp amatrix.cpp adatastructures.cpp)

# add_library(alinalg alinalg.cpp)
# 
# add_library(aoutput aoutput.cpp)
# target_link_libraries(aoutput amesh2dh)
# 
# add_library(amesh2dh amesh2dh.cpp)
# target_link_libraries(amesh2dh adatastructures)
# 
# add_library(amatrix amatrix.cpp)
# 
# add_library(adatastructures adatastructures.cpp)

# for the final executable(s)
add_subdirectory(utilities)

add_executable(fvense explicitsolver.cpp)
target_link_libraries(fvense fvens)

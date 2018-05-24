cmake_minimum_required(VERSION 2.8)

# The environment variable EIGEN_DIR needs to be set to the path of the Eigen top-level directory.
# The env variables PETSC_DIR and PETSC_ARCH need to be set.
# The environment variable BLASTED_DIR may be set to the root of the BLASTed linear algebra library.
# THe env variable BLASTED_BIN_DIR may be set to the root of the BLASTed build directory.

# Pass -DBOOST_ROOT=<path-to-Boost-root-directory> if Boost is not present in a default directory
# Pass -DWITH_BLASTED=1 to compile with BLASTed preconditioning support
# Pass -DNOOMP=1 to compile without OpenMP
# Pass -DSSE=1 to compile with SSE 4.2 instructions; ignored when compiling for KNC.
# Pass -DAVX=1 to compile with AVX instructions.
# Pass -DSKYLAKE=1 to compile with AVX-512 instructions for Xeon Skylake CPUs.
# Pass -DKNL=1 to compile for Xeon Phi Knights Landing.
# Pass -DMICKNC=1 to compile for Xeon Phi Knights Corner.
# Pass -DPROFILE=1 for profiling.
# Pass -DNOTAGS=1 to NOT generate tags for navigating code, otherwise, exuberent-ctags is needed.

project (fvens)

set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror")
set (CMAKE_CXX_FLAGS_DEBUG " ")
set (CMAKE_CXX_FLAGS_RELEASE " ")

# to compile with OpenMP
if(NOOMP)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hnoomp")
	endif()
	message(STATUS "Compiling without OpenMP")
else()
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		message(STATUS "Compiling with OpenMP for GNU")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
		message(STATUS "Compiling with OpenMP for Intel")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
		message(STATUS "Compiling with OpenMP for Cray")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-loop-inline-vec-optimized-missed=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=a")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")
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
	
	# Select hardware-specific optimizations
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
	elseif(SKYLAKE)
		if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx512")
		elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=skylake-avx512")
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for Skylake")
		endif()
	elseif(KNL)
		if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xmic-avx512")
		elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for KNL")
		endif()
	endif()

endif()

# profiling
if(PROFILE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# Boost
if(BOOST_ROOT)
	include_directories(${BOOST_ROOT})
endif()

# Eigen
include_directories($ENV{EIGEN_DIR})

# BLASTed
if(WITH_BLASTED)
	include_directories($ENV{BLASTED_DIR}/include)
	set(BLASTED_LIB "$ENV{BLASTED_BIN_DIR}/lib/libblasted_petsc.so")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BLASTED=1")
	message(STATUS "Building with BLASTed: ${BLASTED_LIB}")
endif()

# PETSc
find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
message(STATUS "Building with PETSc found at ${PETSC_LIB}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_PETSC=1")

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG -DNDEBUG")

string(TOLOWER ${CMAKE_BUILD_TYPE} LCASE_CMAKE_BUILD_TYPE)
if(LCASE_CMAKE_BUILD_TYPE MATCHES "debug")
    message(STATUS "Debug build")
endif()
if(LCASE_CMAKE_BUILD_TYPE MATCHES "release")
	message(STATUS "Release build")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src)

# libraries to be compiled
add_subdirectory(src)

# Tags
if(NOT NOTAGS)
	set_source_files_properties(tags PROPERTIES GENERATED true)
	add_custom_target ( tags
		COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} 
	  )

	find_program(CTAGS_PATH ctags)
	if(CTAGS_PATH)
		message(STATUS "Found ctags: ${CTAGS_PATH}")
		#add_dependencies (fvens_steady tags)
	else()
		message(STATUS "ctags not found! Tags will not be generated.")
	endif(CTAGS_PATH)
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarking
add_subdirectory(benchmarking)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if(WITH_BLASTED AND NOT NOOMP)

	#add_library(threads_async_testing threads_async_tests.cpp)
	#target_link_libraries(threads_async_testing fvens_base ${PETSC_LIB} ${BLASTED_LIB})

	#add_executable(bench_threads_async threads_async.cpp)
	#target_link_libraries(bench_threads_async threads_async_testing)

else()
	message(STATUS "Not building thread async benchmark program.")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if(WITH_BLASTED AND NOT NOOMP)

	add_executable(bench_thread_async threads_async_speedup.cpp)
	target_link_libraries(bench_thread_async fvens_base ${PETSC_LIB} ${BLASTED_LIB})

else()
	message(STATUS "Not building thread async benchmark program.")
endif()

#! /bin/bash

# Execute an async threads-speedup benchmark and check the report generated

@CMAKE_BINARY_DIR@/bench_threads_async $@

@CMAKE_CURRENT_BINARY_DIR@/check_bench_output @CMAKE_CURRENT_BINARY_DIR@/2dcyl
retval=$?

if [ $retval -eq 0 ]
then
	return 0
else
	return 2
fi

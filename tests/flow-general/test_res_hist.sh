#! /bin/bash

## Executes an explicit solve and check the residual history generated
# Needs the option `--log_file_prefix` to be set to the path of the generated convergence history file.

orig_options=$@

# Find the log file option

while [ $# -gt 0 ]
do
key="$1"

case $key in
    --log_file_prefix)
    logfileprefix="$2"
    shift # past argument
    shift # past value
    ;;
    *) # unknown option
	shift
	;;
esac
done

@SEQEXEC@ @THREADOPTS@ @SEQTASKS@ @CMAKE_BINARY_DIR@/tests/e_testflow $orig_options
retval=$?
if [ $retval -ne 0 ]; then
	echo 'Test failed!!!'
	exit $retval
fi

@CMAKE_CURRENT_BINARY_DIR@/runtest_res_hist $logfileprefix

exit

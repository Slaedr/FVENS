/** \file threads_async.cpp
 * \brief Carries out benchmarking tests related to thread-parallel asynchronous preconditioning
 *
 * Command-line or PETSc options file parameters:
 * * -benchmark_type ["speedup_sweeps": Study speed-up obtained from different numbers of async sweeps 
 *     with a fixed number of threads, "none"]
 * * -benchmark_num_repeat [integer] Number of times to repeat the benchmark and average the results
 * * -threads_sequence [integer array] The number of threads to use for the testing; only the first
 *     entry of this array is considered for the 'speedup_sweeps' test.
 * * -async_sweep_sequence [integer array] The number of asynchronous sweeps to run test(s) with; when
 *     the building and application of the preconditioner are both asynchronous, the number of
 *     build sweeps equals the number of application sweeps.
 *
 * \author Aditya Kashi
 * \date 2018-03
 */

#include <iostream>
#include <fstream>
#include <string>
#include <petscsys.h>

#include "../src/autilities.hpp"
#include "threads_async_tests.hpp"

using namespace acfd;
using namespace benchmark;

#define ARR_LEN 10

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Carries out benchmarking tests related to thread-parallel \
		asynchronous preconditioning\n\
		Arguments needed: FVENS control file,\n optionally PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	
	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);

	std::ofstream outf;
	open_file_toWrite(opts.logfile, outf);

	const std::string testtype = parsePetscCmd_string("-benchmark_type", 20);
	const int bnrepeat = parsePetscCmd_int("-benchmark_num_repeat");

	if(testtype == "speedup_sweeps")
	{
		const std::vector<int> threadseq = parsePetscCmd_intArray("-threads_sequence", ARR_LEN);
		const std::vector<int> sweepseq  = parsePetscCmd_intArray("-async_sweep_sequence", ARR_LEN);
		ierr = test_speedup_sweeps(opts, bnrepeat, threadseq[0], sweepseq, outf);
		CHKERRQ(ierr);
	}
	else {
		std::cout << "No benchmark selected.\n";
	}

	outf.close();
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

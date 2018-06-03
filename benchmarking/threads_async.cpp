/** \file threads_async.cpp
 * \brief Carries out benchmarking tests related to thread-parallel asynchronous preconditioning
 *
 * Command-line or PETSc options file parameters:
 * * -benchmark_type ["speedup_sweeps": Study speed-up obtained from different numbers of async sweeps 
 *     with a fixed number of threads, "none"]
 * * -benchmark_num_repeat [integer] Number of times to repeat the benchmark and average the results
 * * -threads_sequence [integer array] The number of threads to use for the testing; only the first
 *     entry of this array is considered for the 'speedup_sweeps' test.
 * * -async_build_sweep_sequence [integer array] The number of asynchronous preconditioner build sweeps 
 *     to run test(s) with
 * * -async_apply_sweep_sequence [integer array] The number of asynchronous apply sweeps.
 *     The length of this array must be same as that of build sweeps array.
 *
 * \author Aditya Kashi
 * \date 2018-03
 */

#include <iostream>
#include <fstream>
#include <string>
#include <petscsys.h>

#include "utilities/aoptionparser.hpp"
#include "utilities/aerrorhandling.hpp"
#include "threads_async_tests.hpp"

using namespace acfd;
using namespace benchmark;
namespace po = boost::program_options;
using namespace std::literals::string_literals;

#define ARR_LEN 10

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Carries out benchmarking tests related to thread-parallel \
		asynchronous preconditioning\n\
		Arguments needed: FVENS control file,\n optionally PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	
	po::options_description desc
		("Utility for bencharking BLASTed asynchronous preconditioners.\n"s
		 + " The first argument is the input control file name.\n"
		 + "Further options");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	std::ofstream outf;
	open_file_toWrite(opts.logfile+".logb", outf);

	const std::string testtype = parsePetscCmd_string("-benchmark_type", 20);
	const int bnrepeat = parsePetscCmd_int("-benchmark_num_repeat");

	// write a message to stdout about which preconditioner is being used -
	//  could be useful for reading the Moab log file
	std::string prec = parsePetscCmd_string("-blasted_pc_type", 10);
	std::cout << ">>> Benchmark " << testtype << ", preconditioner " << prec << std::endl;

	if(testtype == "speedup_sweeps")
	{
		const std::vector<int> threadseq = parsePetscCmd_intArray("-threads_sequence", ARR_LEN);
		const std::vector<int> bswpseq  = parsePetscCmd_intArray("-async_build_sweep_sequence",
		                                                         ARR_LEN);
		const std::vector<int> aswpseq  = parsePetscCmd_intArray("-async_apply_sweep_sequence",
		                                                         ARR_LEN);
		fvens_throw(bswpseq.size() != aswpseq.size(),
		            "There must be an apply sweep for each build sweep requested and vice-versa!");

		ierr = test_speedup_sweeps(opts, bnrepeat, threadseq, bswpseq, aswpseq, outf);
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

/** \file threads_async.cpp
 * \brief Carries out benchmarking tests related to thread-parallel asynchronous preconditioning
 *
 * Command-line or PETSc options file parameters:
 * * -benchmark_type ["speedup_sweeps": Study speed-up obtained from different numbers of async sweeps 
 *     with a fixed number of threads, "none"]
 * * -benchmark_num_repeat [integer] Number of times to repeat the benchmark and average the results
 * * -benchmark_output_file [string] Path to which to write out the benchmark report
 * * -benchmark_threads_sequence [integer array] The number of threads to use for the testing
 * * -benchmark_base_threads [integer] Number of threads to use for base run
 * * -async_build_sweep_sequence [integer array] The number of asynchronous preconditioner build sweeps 
 *     to run test(s) with
 * * -async_apply_sweep_sequence [integer array] The number of asynchronous apply sweeps.
 *     The length of this array must be same as that of build sweeps array.
 * * -async_base_build_sweeps
 * * -async_base_apply_sweeps
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
#include "utilities/casesolvers.hpp"
#include "threads_async_tests.hpp"

using namespace fvens;
using namespace benchmark;
namespace po = boost::program_options;
using namespace std::literals::string_literals;

#define ARR_LEN 10
#define PATH_LEN 100

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
	const std::string outfilename = parsePetscCmd_string("-benchmark_output_file", PATH_LEN);
	open_file_toWrite(outfilename, outf);

	const std::string testtype = parsePetscCmd_string("-benchmark_type", 20);
	const int bnrepeat = parsePetscCmd_int("-benchmark_num_repeat");

	// write a message to stdout about which preconditioner is being used -
	//  could be useful for reading the Moab log file
	std::string prec = parsePetscCmd_string("-blasted_pc_type", 10);
	std::cout << ">>> Benchmark " << testtype << ", preconditioner " << prec << std::endl;

	if(testtype == "speedup_sweeps")
	{
		SpeedupSweepsConfig config;
		config.threadSeq = parsePetscCmd_intArray("-benchmark_threads_sequence", ARR_LEN);
		config.buildSwpSeq = parsePetscCmd_intArray("-async_build_sweep_sequence", ARR_LEN);
		config.applySwpSeq  = parsePetscCmd_intArray("-async_apply_sweep_sequence", ARR_LEN);
		config.basethreads = parsePetscCmd_int("-benchmark_base_threads");
		config.basebuildsweeps = parsePetscCmd_int("-async_base_build_sweeps");
		config.baseapplysweeps = parsePetscCmd_int("-async_base_apply_sweeps");
		fvens_throw(buildSwpSeq.size() != applySwpSeq.size(),
		            "There must be an apply sweep for each build sweep requested and vice-versa!");

		ierr = test_speedup_sweeps(opts, bnrepeat, config, outf);
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

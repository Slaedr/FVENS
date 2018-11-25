/** \file flowpseudotime.cpp
 * \brief Driver function for tests related to the steady pseudo-temporal solution process 
 * \author Aditya Kashi
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <petscvec.h>

#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/casesolvers.hpp"

using namespace fvens;
namespace po = boost::program_options;
using namespace std::literals::string_literals;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

	po::options_description desc
		("FVENS functional convergence test.\n"s
		 + " The first argument is the input control file name.\n"
		 + "Further options");
	desc.add_options()("test_type", po::value<std::string>(), "Type of test: 'exception_nanorinf' \
for testing detection of NaN or inf during nonlinear sovlve");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);
	const UMesh2dh<a_real> m = constructMeshFlow(opts, "");
	SteadyFlowCase case1(opts);

	// solution vector
	Vec u;
	ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);

	//std::string testchoice = parsePetscCmd_string("-test_type", 100);
	std::string testchoice = cmdvars["test_type"].as<std::string>();

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	if(testchoice == "exception_nanorinf") {
		try {
			ierr = case1.run(m, u);
			CHKERRQ(ierr);
		}
		catch (Numerical_error& e) {
			return 0;
		}
		return -1;
	}

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

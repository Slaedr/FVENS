/** \file flow_solve.cpp
 * \brief Driver function for testing successful solution of one case, with no output files.
 * \author Aditya Kashi
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <petscvec.h>

#include "linalg/alinalg.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/casesolvers.hpp"

using namespace fvens;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

	po::options_description desc
		(std::string("FVENS options: The first argument is the input control file name.\n")
		 + "Further options");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	// solution vector
	Vec u;

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	SteadyFlowCase case1(opts);
	ierr = case1.run("", &u); CHKERRQ(ierr);

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

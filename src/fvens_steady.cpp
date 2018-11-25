#include <iostream>
#include <string>
#include <petscvec.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

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

	// First set up command line options parsing

	po::options_description desc
		(std::string("FVENS - Finite volume Euler and Navier-Stokes solver")
		 + "\n The first argument must be the control file to use. Further options");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	// Mesh
	const UMesh2dh<a_real> m = constructMeshFlow(opts, "");

	// solution vector
	Vec u;
	ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	SteadyFlowCase case1(opts);
	case1.run_output(true, true, m, u);

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

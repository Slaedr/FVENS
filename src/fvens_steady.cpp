#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "linalg/alinalg.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/afactory.hpp"
#include "utilities/casesolvers.hpp"
#include "spatial/aoutput.hpp"
#include "ode/aodesolver.hpp"
#include "mesh/ameshutils.hpp"

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

using namespace amat;
using namespace acfd;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

	// First set up command line options parsing

	po::options_description desc
		(std::string("FVENS options: The first argument is always the input control file name.\n")
		 + "Further options");
	desc.add_options()
		("help", "help message")
		("mesh_file", po::value<std::string>(),
		 "Mesh file to solve the problem on; overrides the corresponding option in the control file");

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);

	// solution vector
	Vec u;

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	SteadyFlowCase case1(opts);
	case1.run_output("", true, &u);

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

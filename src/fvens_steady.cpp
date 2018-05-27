#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>

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

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

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

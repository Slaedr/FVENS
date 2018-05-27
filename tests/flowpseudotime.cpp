/** \file flowpseudotime.cpp
 * \brief Driver function for tests related to the steady pseudo-temporal solution process 
 * \author Aditya Kashi
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>

#include "linalg/alinalg.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/afactory.hpp"
#include "utilities/casesolvers.hpp"
#include "utilities/aerrorhandling.hpp"
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
	SteadyFlowCase case1(opts);

	// solution vector
	Vec u;

	std::string testchoice = parsePetscCmd_string("-test_type", 100);

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	if(testchoice == "exception_nanorinf") {
		try {
			ierr = case1.run("", &u);
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

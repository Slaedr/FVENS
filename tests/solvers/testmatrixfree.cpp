#undef NDEBUG

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

	po::options_description desc ("Test for matrix-free solver");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	const UMesh2dh<a_real> m = constructMesh(opts, "");
	const FlowFV_base<a_real> *const spatial = createFlowSpatial(opts, m);

	Vec u;
	ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);

	SteadyFlowCase case1(opts);
	TimingData td1 = case1.execute_main(spatial, u);
	if(!td1.converged)
		throw Tolerance_error("Mat-based solve did not converge to specified tolerance!");

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	ierr = PetscOptionsSetValue(NULL, "-matrix_free_jacobian", ""); CHKERRQ(ierr);
	ierr = PetscOptionsSetValue(NULL, "-matrix_free_difference_step", "1e-6"); CHKERRQ(ierr);

	SteadyFlowCase case2(opts);
	ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);
	TimingData td2 = case2.execute_main(spatial, u);
	if(!td2.converged)
		throw Tolerance_error("Mat-free solve did not converge to specified tolerance!");

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	delete spatial;

	std::cout << "Matrix-based iterations = " << td1.num_timesteps << std::endl;
	std::cout << "Matrix-free iterations = " << td2.num_timesteps << std::endl;

	assert(abs(td1.num_timesteps-td2.num_timesteps) <= 1);

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

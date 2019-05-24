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
#include "utilities/mpiutils.hpp"

#define REGR_DEFAULT_TOL 1e-13

using namespace fvens;
namespace po = boost::program_options;

static void test_regression(const po::variables_map& cmdvars, const FlowSolutionFunctionals fnls);

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	po::options_description desc
		(std::string("FVENS options: The first argument is the input control file name.\n")
		 + "Further options");

	desc.add_options()
		("regression_test", po::value<bool>(), "1 to compare with a previous result, 0 otherwise")
		("regression_file", po::value<std::string>(), "File name containing regression data")
		("regression_tol", po::value<double>(), "Tolerance for regression test");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	// Mesh
	const UMesh<freal,NDIM> m = constructMeshFlow(opts, "");
	// solution vector
	Vec u;
	ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);

	// solve case - constructs (creates) u, computes the solution and stores the solution in it
	SteadyFlowCase case1(opts);
	//ierr = case1.run(m, u); CHKERRQ(ierr);
	const FlowSolutionFunctionals fnls = case1.run_output(false, false, m, u);

	std::cout << "Output:\n" << std::setprecision(15)
	          << "h = " << fnls.meshSizeParameter
	          << ", CL = " << fnls.CL << ",  CDp = " << fnls.CDp << ",  CDsf = " << fnls.CDsf
	          << std::endl;

	test_regression(cmdvars, fnls);

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	if(mpirank == 0)
		std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	if(mpirank == 0)
		std::cout << "\n--------------- End --------------------- \n\n";
	return ierr;
}

void test_regression(const po::variables_map& cmdvars, const FlowSolutionFunctionals fnls)
{
	const bool run_regr = cmdvars.count("regression_test") ?
		cmdvars["regression_test"].as<bool>() : false;
	const std::string regr_file = cmdvars.count("regression_test") ?
		cmdvars["regression_file"].as<std::string>() : "";
	const double regr_tol = cmdvars.count("regression_tol") ?
		cmdvars["regression_tol"].as<double>() : REGR_DEFAULT_TOL;

	if(run_regr)
	{
		std::cout << "  Running regression test with file " << regr_file << std::endl;

		std::ifstream rfile;
		open_file_toRead(regr_file, rfile);

		double CDp, CL, CDsf;
		rfile >> CL >> CDp >> CDsf;

		rfile.close();

		if(std::abs(CL-fnls.CL)/std::abs(CL) > regr_tol) {
			std::cout << "  CL error = " << std::abs(CL-fnls.CL)/std::abs(CL) << std::endl;
			throw std::runtime_error("CL does not match!");
		}
		if(std::abs(CDp-fnls.CDp)/std::abs(CDp) > regr_tol) {
			std::cout << "  CDp error = " << std::abs(CDp-fnls.CDp)/std::abs(CDp) << std::endl;
			throw std::runtime_error("CDp does not match!");
		}
		// if(std::abs(CDsf-fnls.CDsf)/std::abs(CDsf) > regr_tol)
		// 	throw std::runtime_error("CDsf does not match!");
	}
}

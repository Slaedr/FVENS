/** \file flow_clcd_conv.cpp
 * \brief Spatial convergence tests for target functionals
 * \author Aditya Kashi
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <petscvec.h>

#include "utilities/aerrorhandling.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/casesolvers.hpp"

using namespace fvens;
namespace po = boost::program_options;
using namespace std::literals::string_literals;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Test for convergence in lift and drag.\n\
		Arguments needed: FVENS control file, PETSc options file with -options_file,\n\
		-number_of_meshes, -test_type and -exact_solution_file.";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	po::options_description desc
		("FVENS functional convergence test.\n"s
		 + " The first argument is the input control file name.\n"
		+ "Further options");
	desc.add_options()("number_of_meshes", po::value<int>(),
	                   "Number of meshes for grid convergence test");
	desc.add_options()("test_type", po::value<std::string>(),
	                   "Type of test: 'CL', 'CDP' or 'CDSF' for lift, pressure drag or \
skin-friction drag respectively");
	desc.add_options()("exact_solution_file", po::value<std::string>(),
	                   "Location of file containing the exact solution for the desired target functional");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	// Get number of meshes
	//const int nmesh = parsePetscCmd_int("-number_of_meshes");
	const int nmesh = cmdvars["number_of_meshes"].as<int>();

	// get test type - CL, CDP or CDSF
	//const std::string test_type = parsePetscCmd_string("-test_type", 20);
	const std::string test_type = cmdvars["test_type"].as<std::string>();

	// get the locaiton of the file containing the exact CL, CDp and CDsf
	//const std::string exf = parsePetscCmd_string("-exact_solution_file", 100);
	const std::string exf = cmdvars["exact_solution_file"].as<std::string>();

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// read exact calues
	a_real ex_CL, ex_CDp, ex_CDsf;
	std::ifstream fexact;
	fexact.open(exf);
	if(!fexact) {
		throw std::runtime_error("! Could not open exact soln file!");
	}
	fexact >> ex_CL >> ex_CDp >> ex_CDsf;
	fexact.close();
	std::cout << "Exact values of CL, CDp and CDsf are " << ex_CL << " " << ex_CDp << " " << ex_CDsf
		<< std::endl;

	// Read control file

	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	SteadyFlowCase case1(opts);
	
	std::vector<double> lh(nmesh), clerrors(nmesh), cdperrors(nmesh), cdsferrors(nmesh),
		clslopes(nmesh-1), cdpslopes(nmesh-1), cdsfslopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++)
	{
		// Mesh
		std::string meshsuffix = std::to_string(imesh) + ".msh";
		const UMesh2dh<a_real> m = constructMeshFlow(opts, meshsuffix);
		
		// solution vector
		Vec u;
		ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);
		
		FlowSolutionFunctionals fnls;
		try {
			fnls = case1.run_output(false, false, m, u);
		} catch (Tolerance_error& e) {
			std::cout << e.what() << std::endl;
		}

		std::cout << "CL Cdp CDsf = " << fnls.CL << " " << fnls.CDp << " " << fnls.CDsf << std::endl;
		
		lh[imesh] = log10(fnls.meshSizeParameter);
		clerrors[imesh]   = log10(std::abs( std::abs(fnls.CL)  -ex_CL ));
		cdperrors[imesh]  = log10(std::abs( std::abs(fnls.CDp) -ex_CDp ));
		cdsferrors[imesh] = log10(std::abs( std::abs(fnls.CDsf)-ex_CDsf ));
		if(imesh > 0) {
			clslopes[imesh-1] = (clerrors[imesh]-clerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);
			cdpslopes[imesh-1] = (cdperrors[imesh]-cdperrors[imesh-1])/(lh[imesh]-lh[imesh-1]);
			cdsfslopes[imesh-1] = (cdsferrors[imesh]-cdsferrors[imesh-1])/(lh[imesh]-lh[imesh-1]);
		}

		if(imesh > 0) {
			std::cout << ">> Orders = \n" ;
			std::cout << "CL:   " << clslopes[imesh-1] << std::endl;
			std::cout << "CDp:   " << cdpslopes[imesh-1] << std::endl;
			std::cout << "CDsf:  " << cdsfslopes[imesh-1] << std::endl;
		}
		std::cout << std::endl;

		ierr = VecDestroy(&u); CHKERRQ(ierr);
	}
	
	std::cout << "> Orders = \n" ;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "CL:   " << clslopes[i] << std::endl;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "CDp:   " << cdpslopes[i] << std::endl;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "CDsf:  " << cdsfslopes[i] << std::endl;
	
	int passed = 0;
	if(test_type == "CL") 
	{
		if(clslopes[nmesh-2] <= 2.5 && clslopes[nmesh-2] >= 1.9)
			passed = 1;
	}
	else if(test_type == "CDP") 
	{
		if(cdpslopes[nmesh-2] <= 2.5 && cdpslopes[nmesh-2] >= 1.9)
			passed = 1;
	}
	else if(test_type == "CDSF") 
	{
		if(cdsfslopes[nmesh-2] <= 1.5 && cdsfslopes[nmesh-2] >= 0.95)
			passed = 1;
	}
	else {
		std::cout << "Unrecognized test!\n";
		std::abort();
	}

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return !passed;
}

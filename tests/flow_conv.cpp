#include <iostream>
#include <iomanip>
#include <string>
#include <petscvec.h>

#include "utilities/aerrorhandling.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/casesolvers.hpp"

using namespace fvens;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Test for convergence in entropy.\n\
		Arguments needed: FVENS control file, PETSc options file with -options_file,\n\
		and -number_of_meshes";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	po::options_description desc
		(std::string("FVENS Conv options: The first argument is the input control file name.\n")
		 + "Further options");
	desc.add_options()("number_of_meshes", po::value<int>(),
	                   "Number of meshes for grid convergence test");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	// Get number of meshes
	//const int nmesh = parsePetscCmd_int("-number_of_meshes");
	const int nmesh = cmdvars["number_of_meshes"].as<int>();

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	SteadyFlowCase case1(opts);

	std::vector<double> lh(nmesh), lerrors(nmesh), slopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++) 
	{
		// Mesh file suffix
		std::string meshsuffix = std::to_string(imesh) + ".msh";
		//Mesh
		const UMesh2dh<a_real> m = constructMesh(opts, meshsuffix);

		// solution vector
		Vec u;
		ierr = initializeSystemVector(opts, m, &u); CHKERRQ(ierr);
		
		FlowSolutionFunctionals fnls;
		try {
			fnls = case1.run_output(false, false, m, u);
		} catch(Numerical_error& e) {
			std::cout << e.what() << std::endl;
		}

		const a_real h = fnls.meshSizeParameter;
		const a_real err = fnls.entropy;
		std::cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << std::endl;
		lh[imesh] = log10(h);
		lerrors[imesh] = log10(err);
		if(imesh > 0)
			slopes[imesh-1] = (lerrors[imesh]-lerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);

		ierr = VecDestroy(&u); CHKERRQ(ierr);
	}
	
	std::cout << ">> Spatial orders = \n" ;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "   " << slopes[i] << std::endl;
	
	int passed = 0;
	if(opts.gradientmethod == "LEASTSQUARES") 
	{
		// the lower limit is chosen from experience
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.65)
			passed = 1;
	}
	else if(opts.gradientmethod == "GREENGAUSS") 
	{
		// the lower limit is chosen from experience
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.65)
			passed = 1;
	}

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return !passed;
}

#include <iostream>
#include <iomanip>
#include <string>
#include <petscvec.h>

#include "utilities/aoptionparser.hpp"
#include "utilities/casesolvers.hpp"

using namespace amat;
using namespace acfd;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Test for convergence in entropy.\n\
		Arguments needed: FVENS control file, PETSc options file with -options_file,\n\
		and -number_of_meshes";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	// Get number of meshes
	const int nmesh = parsePetscCmd_int("-number_of_meshes");

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);

	std::vector<double> lh(nmesh), lerrors(nmesh), slopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++) 
	{
		// solution vector
		Vec u;
		
		// Mesh file suffix
		std::string meshs = std::to_string(imesh) + ".msh";

		const FlowSolutionFunctionals fnls { steadyCase_output(opts, meshs, &u, false) };

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
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.6)
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

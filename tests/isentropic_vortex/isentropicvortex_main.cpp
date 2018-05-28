#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>

#include "../src/alinalg.hpp"
#include "../src/autilities.hpp"
#include "../src/aoutput.hpp"
#include "../src/aodesolver.hpp"
#include "../src/afactory.hpp"
#include "../src/ameshutils.hpp"
#include "isentropicvortex.hpp"

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

using namespace amat;
using namespace acfd;
using namespace fvens_tests;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	// Get number of meshes
	PetscBool set = PETSC_FALSE;
	int nmesh = 0;
	ierr = PetscOptionsGetInt(NULL, NULL, "-number_of_meshes", &nmesh, &set); CHKERRQ(ierr);
	if(!set) {
		ierr = -1;
		throw "Need number of meshes!";
	}

	// Read control file

	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);

	UnsteadyFlowCase case1(opts);
	
	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);

	// numerics for main solver
	//const FlowNumericsConfig nconfmain {opts.invflux, opts.invfluxjac,
	//	opts.gradientmethod, opts.limiter, opts.order2};
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);

	// simpler numerics for startup
	const FlowNumericsConfig nconfstart = firstorder_spatial_numerics_config(opts);
	
	const SteadySolverConfig maintconf {
		opts.lognres, opts.logfile+".tlog",
		opts.initcfl, opts.endcfl, opts.rampstart, opts.rampend,
		opts.tolerance, opts.maxiter,
	};

	const SteadySolverConfig starttconf {
		opts.lognres, opts.logfile+"-init.tlog",
		opts.firstinitcfl, opts.firstendcfl, opts.firstrampstart, opts.firstrampend,
		opts.firsttolerance, opts.firstmaxiter,
	};

	// Read isen-vortex control params
	std::ifstream infile(argv[2]);
	std::string dum;
	std::array<a_real,2> vcentre;
	a_real strength, clength, sigma;
	infile >> dum; infile >> vcentre[0] >> vcentre[1];
	infile >> dum; infile >> strength;
	infile >> dum; infile >> clength;
	infile >> dum; infile >> sigma;
	infile.close()

	const IsenVortexConfig ivconf {opts.gamma, opts.Minf, vcentre, strength,
			clength, sigma, opts.alpha};
	const IsentropicVortexProblem isen(ivconf);

	std::vector<double> lh(nmesh), lerrors(nmesh), slopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++) {
		
		std::string meshi = opts.meshfile + std::to_string(imesh) + ".msh";
		// Set up mesh

		UMesh2dh m;
		m.readMesh(meshi);
		CHKERRQ(preprocessMesh(m));
		m.compute_periodic_map(opts.periodic_marker, opts.periodic_axis);

		std::cout << "\n***\n";

		std::cout << "Setting up main spatial scheme.\n";
		const Spatial<NVARS> *const prob = create_const_flowSpatialDiscretization(&m, pconf, nconfmain);

		Vec u, uexact;
		ierr = VecCreateSeq(PETSC_COMM_SELF, m->gnelem()*4, &u); CHKERRQ(ierr);
		ierr = VecDuplicate(u, &uexact); CHKERRQ(ierr);

		// get initial and exact solution vectors
		double *uarr, *uexarr;
		ierr = VecGetArray(u, &uarr); CHKERRQ(ierr);
		ierr = VecGetArray(uexact, &uexarr); CHKERRQ(ierr);
		isen.getInitialConditionAndExactSolution(m, opts.final_time, uarr, uexarr);
		ierr = VecRestoreArray(u, &uarr); CHKERRQ(ierr);
		ierr = VecRestoreArray(uexact, &uexarr); CHKERRQ(ierr);

		ierr = case1.execute(prob, u); CHKERRQ(ierr);

		std::cout << "***\n";
		
		a_real err;
		// get the FlowFV to compute the entropy error
		const FlowFV_base* fprob = reinterpret_cast<const FlowFV_base*>(prob);
		err = fprob->compute_entropy_cell(u);
		const double h = 1.0/sqrt(m.gnelem());
		std::cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << std::endl;
		lh[imesh] = log10(h);
		lerrors[imesh] = log10(err);
		if(imesh > 0)
			slopes[imesh-1] = (lerrors[imesh]-lerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);


		delete starttime;
		delete time;
		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
#ifdef USE_BLASTED
		destroyBlastedDataVec(&bctx);
#endif
		ierr = MatDestroy(&M); CHKERRQ(ierr);
		if(mf_flg) {
			ierr = MatDestroy(&A); 
			CHKERRQ(ierr);
		}

		delete prob;
		delete startprob;
	}
	
	std::cout << ">> Spatial orders = \n" ;
	for(int i = 0; i < nmesh-1; i++)
		std::cout << "   " << slopes[i] << std::endl;
	
	int passed = 0;
	if(opts.gradientmethod == "LEASTSQUARES") 
	{
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.8)
			passed = 1;
	}
	else if(opts.gradientmethod == "GREENGAUSS") 
	{
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.65)
			passed = 1;
	}

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return !passed;
}

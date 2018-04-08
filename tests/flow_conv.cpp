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
	
	// physical configuration
	const FlowPhysicsConfig pconf {
		opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr, opts.alpha,
		opts.viscsim, opts.useconstvisc,
		opts.isothermalwall_marker, opts.adiabaticwall_marker, opts.isothermalpressurewall_marker,
		opts.slipwall_marker, opts.farfield_marker, opts.inout_marker,
		opts.extrap_marker, opts.periodic_marker,
		opts.twalltemp, opts.twallvel, opts.adiawallvel, opts.tpwalltemp, opts.tpwallvel
	};

	// numerics for main solver
	const FlowNumericsConfig nconfmain {opts.invflux, opts.invfluxjac,
		opts.gradientmethod, opts.limiter, opts.order2};

	// simpler numerics for startup
	const FlowNumericsConfig nconfstart {opts.invflux, opts.invfluxjac, "NONE", "NONE", false};
	
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

		std::cout << "\nSetting up spatial scheme for the initial guess.\n";
		const Spatial<NVARS> *const startprob
			= create_const_flowSpatialDiscretization(&m, pconf, nconfstart);

		std::cout << "\n***\n";

		/* NOTE: Since the "startup" solver (meant to generate an initial solution) and the "main" solver
		 * have the same number of unknowns and use first-order Jacobians, we have just one set of
		 * solution vector, Jacobian matrix, preconditioning matrix and KSP solver.
		 */

		// solution vector
		Vec u;

		// Initialize Jacobian for implicit schemes
		Mat M;
		ierr = setupSystemMatrix<NVARS>(&m, &M); CHKERRQ(ierr);
		ierr = MatCreateVecs(M, &u, NULL); CHKERRQ(ierr);

		// setup matrix-free Jacobian if requested
		Mat A;
		MatrixFreeSpatialJacobian<NVARS> mfjac;
		PetscBool mf_flg = PETSC_FALSE;
		ierr = PetscOptionsHasName(NULL, NULL, "-matrix_free_jacobian", &mf_flg); CHKERRQ(ierr);
		if(mf_flg) {
			std::cout << " Allocating matrix-free Jac\n";
			ierr = setup_matrixfree_jacobian<NVARS>(&m, &mfjac, &A); 
			CHKERRQ(ierr);
		}

		// initialize solver
		KSP ksp;
		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		if(mf_flg) {
			ierr = KSPSetOperators(ksp, A, M); 
			CHKERRQ(ierr);
		}
		else {
			ierr = KSPSetOperators(ksp, M, M); 
			CHKERRQ(ierr);
		}
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// set up time discrization

		SteadySolver<NVARS> * starttime=nullptr, * time=nullptr;

		if(opts.timesteptype == "IMPLICIT")
		{
			if(opts.usestarter != 0) {
				starttime = new SteadyBackwardEulerSolver<NVARS>(startprob, starttconf, ksp);
				std::cout << "Set up backward Euler temporal scheme for initialization solve.\n";
			}

		}
		else
		{
			if(opts.usestarter != 0) {
				starttime = new SteadyForwardEulerSolver<NVARS>(startprob, u, starttconf);
				std::cout << "Set up explicit forward Euler temporal scheme for startup solve.\n";
			}
		}

		// Ask the spatial discretization context to initialize flow variables
		startprob->initializeUnknowns(u);

		// setup BLASTed preconditioning if requested
#ifdef USE_BLASTED
		Blasted_data_vec bctx = newBlastedDataVec();
		if(opts.timesteptype == "IMPLICIT") {
			ierr = setup_blasted<NVARS>(ksp,u,startprob,bctx); CHKERRQ(ierr);
		}
#endif

		std::cout << "\n***\n";

		// computation

		if(opts.usestarter != 0) {

			mfjac.set_spatial(startprob);

			// solve the starter problem to get the initial solution
			ierr = starttime->solve(u); CHKERRQ(ierr);
		}

		// Reset the KSP - could be advantageous for some types of algebraic solvers
		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		destroyBlastedDataVec(bctx);
		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		if(mf_flg) {
			ierr = KSPSetOperators(ksp, A, M); 
			CHKERRQ(ierr);
		}
		else {
			ierr = KSPSetOperators(ksp, M, M); 
			CHKERRQ(ierr);
		}
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
#ifdef USE_BLASTED
		// this will reset the timing
		bctx = newBlastedDataVec();
		if(opts.timesteptype == "IMPLICIT") {
			ierr = setup_blasted<NVARS>(ksp,u,startprob,bctx); CHKERRQ(ierr);
		}
#endif

		// setup nonlinear ODE solver for main solve
		if(opts.timesteptype == "IMPLICIT")
		{
			time = new SteadyBackwardEulerSolver<4>(prob, maintconf, ksp);
			std::cout << "\nSet up backward Euler temporal scheme for main solve.\n";
		}
		else
		{
			time = new SteadyForwardEulerSolver<4>(prob, u, maintconf);
			std::cout << "\nSet up explicit forward Euler temporal scheme for main solve.\n";
		}

		mfjac.set_spatial(prob);

		// Solve the main problem
		ierr = time->solve(u); CHKERRQ(ierr);

		std::cout << "***\n";
		
		a_real err;
		// get the correct kind of FlowFV - but there's got to be a better way to do this
		if(opts.order2 && opts.useconstvisc) {
			const FlowFV<true,true>* fprob = reinterpret_cast<const FlowFV<true,true>*>(prob);
			err = fprob->compute_entropy_cell(u);
		}
		else if(opts.order2 && !opts.useconstvisc) {
			const FlowFV<true,false>* fprob = reinterpret_cast<const FlowFV<true,false>*>(prob);
			err = fprob->compute_entropy_cell(u);
		}
		else if(!opts.order2 && opts.useconstvisc) {
			const FlowFV<false,true>* fprob = reinterpret_cast<const FlowFV<false,true>*>(prob);
			err = fprob->compute_entropy_cell(u);
		}
		else {
			const FlowFV<false,false>* fprob = reinterpret_cast<const FlowFV<false,false>*>(prob);
			err = fprob->compute_entropy_cell(u);
		}
		const double h = 1.0/sqrt(m.gnelem());
		std::cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << std::endl;
		lh[imesh] = log10(h);
		lerrors[imesh] = log10(err);
		if(imesh > 0)
			slopes[imesh-1] = (lerrors[imesh]-lerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);


		delete starttime;
		delete time;
		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
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
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.6)
			passed = 1;
	}
	else if(opts.gradientmethod == "GREENGAUSS") 
	{
		if(slopes[nmesh-2] <= 2.1 && slopes[nmesh-2] >= 1.7)
			passed = 1;
	}

	std::cout << '\n';
	ierr = PetscFinalize(); CHKERRQ(ierr);
	std::cout << "\n--------------- End --------------------- \n\n";
	return !passed;
}

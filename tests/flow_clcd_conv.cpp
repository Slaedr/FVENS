#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
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

	// get test type - CL, CDP or CDSF
	set = PETSC_FALSE;
	constexpr size_t p_strlen = 10;
	char tt[p_strlen];
	ierr = PetscOptionsGetString(NULL, NULL, "-test_type", tt, p_strlen, &set); CHKERRQ(ierr);
	if(!set) {
		ierr = -1;
		throw "Need test type!";
	}
	const std::string test_type = tt;

	// get the locaiton of the file containing the exact CL, CDp and CDsf
	set = PETSC_FALSE;
	constexpr size_t p_filelen = 100;
	char exf[p_filelen];
	ierr = PetscOptionsGetString(NULL, NULL, "-exact_solution_file", exf, p_filelen, &set); 
	CHKERRQ(ierr);
	if(!set) {
		ierr = -1;
		throw "Need exact CL, CDp and CDsf!";
	}

	// read exact calues
	a_real ex_CL, ex_CDp, ex_CDsf;
	std::ifstream fexact;
	fexact.open(exf);
	if(!fexact) {
		//std::cout << "! Could not open file "<< exf <<" !\n";
		//std::abort();
		throw std::runtime_error("! Could not open exact soln file!");
	}
	fexact >> ex_CL >> ex_CDp >> ex_CDsf;
	fexact.close();
	std::cout << "Exact values of CL, CDp and CDsf are " << ex_CL << " " << ex_CDp << " " << ex_CDsf
		<< std::endl;

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

	std::vector<double> lh(nmesh), clerrors(nmesh), cdperrors(nmesh), cdsferrors(nmesh),
		clslopes(nmesh-1), cdpslopes(nmesh-1), cdsfslopes(nmesh-1);

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
#ifdef USE_BLASTED
		destroyBlastedDataVec(&bctx);
#endif
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

		// Output
	
		MVector umat; umat.resize(m.gnelem(),NVARS);
		const PetscScalar *uarr;
		ierr = VecGetArrayRead(u, &uarr); CHKERRQ(ierr);
		for(a_int i = 0; i < m.gnelem(); i++)
			for(int j = 0; j < NVARS; j++)
				umat(i,j) = uarr[i*NVARS+j];
		ierr = VecRestoreArrayRead(u, &uarr);

		IdealGasPhysics phy(opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr);
		FlowOutput out(&m, prob, &phy, opts.alpha);
	
		MVector output; output.resize(m.gnelem(),NDIM+2);
		std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>> grad;
		grad.resize(m.gnelem());
		prob->getGradients(umat, grad);

		// get Cl, Cdp and Cdsf of the first wall boundary marker only
		std::tuple<a_real,a_real,a_real> fnls 
			{ out.computeSurfaceData(umat, grad, opts.lwalls[0], output)};
		std::cout << "CL Cdp CDsf = " << std::get<0>(fnls) << " " << std::get<1>(fnls) << " " <<
			std::get<2>(fnls) << std::endl;
		
		lh[imesh] = log10(1.0/sqrt(m.gnelem()));   // 2D only
		clerrors[imesh] = log10(std::abs( std::abs(std::get<0>(fnls))-ex_CL ));
		cdperrors[imesh] = log10(std::abs( std::abs(std::get<1>(fnls))-ex_CDp ));
		cdsferrors[imesh] = log10(std::abs( std::abs(std::get<2>(fnls))-ex_CDsf ));
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

		delete prob;
		delete startprob;
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

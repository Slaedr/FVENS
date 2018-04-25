/** \file threads_async_tests.cpp
 * \brief Implementation of tests related to multi-threaded asynchronous preconditioning
 * \author Aditya Kashi
 * \date 2018-03
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include <petscksp.h>

#include "utilities/afactory.hpp"
#include "linalg/alinalg.hpp"
#include "ode/aodesolver.hpp"
#include "mesh/ameshutils.hpp"

#include <blasted_petsc.h>

#include "threads_async_tests.hpp"

namespace benchmark {

using namespace acfd;

/// Set -blasted_async_sweeps in the default Petsc options database and throws if not successful
static void set_blasted_sweeps(const int nbswp, const int naswp);

StatusCode test_speedup_sweeps(const FlowParserOptions& opts, const int numrepeat, const int numthreads,
		const std::vector<int>& sweep_seq, const double sweepratio, std::ofstream& outf)
{
	StatusCode ierr = 0;

	// Set up mesh
	UMesh2dh m;
	m.readMesh(opts.meshfile);
	CHKERRQ(preprocessMesh(m));
	m.compute_periodic_map(opts.periodic_marker, opts.periodic_axis);
	std::cout << "\n***\n";

	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// numerics for main solver
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);
	// simpler numerics for startup
	const FlowNumericsConfig nconfstart {opts.invflux, opts.invfluxjac, "NONE", "NONE", false};

	std::cout << "Setting up main spatial scheme.\n";
	const Spatial<NVARS> *const prob = create_const_flowSpatialDiscretization(&m, pconf, nconfmain);
	std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<NVARS> *const startprob
		= create_const_flowSpatialDiscretization(&m, pconf, nconfstart);
	std::cout << "\n***\n";

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

	SteadySolver<NVARS> * starttime=nullptr, * time=nullptr;

	if(opts.usestarter != 0) {
		starttime = new SteadyBackwardEulerSolver<NVARS>(startprob, starttconf, ksp);
		std::cout << "Set up backward Euler temporal scheme for initialization solve.\n";
	}

	// Ask the spatial discretization context to initialize flow variables
	startprob->initializeUnknowns(u);

	// setup BLASTed preconditioning
	Blasted_data_list bctx = newBlastedDataList();
	ierr = setup_blasted<NVARS>(ksp,u,startprob,bctx); CHKERRQ(ierr);

	std::cout << "\n***\n";

	// starting computation
	omp_set_num_threads(1);
	set_blasted_sweeps(1,1);
	if(opts.usestarter != 0) {

		mfjac.set_spatial(startprob);

		// solve the starter problem to get the initial solution
		ierr = starttime->solve(u); CHKERRQ(ierr);
	}

	// Benchmarking runs

	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);
	if(mpirank == 0) {
		outf << "# Preconditioner wall times #\n# num-cells = " << m.gnelem() << "\n";
	}

	omp_set_num_threads(1);

	TimingData tdata = run_sweeps(startprob, prob, maintconf, 1, 1, &ksp, u, A, M, 
			mfjac, mf_flg, bctx);

	computeTotalTimes(&bctx);
	const double factor_basewtime = bctx.factorwalltime;
	const double apply_basewtime = bctx.applywalltime;
	const double prec_basewtime = bctx.factorwalltime + bctx.applywalltime;

	const int w = 11;
	if(mpirank == 0) {
		outf << "# Base preconditioner wall time = " << prec_basewtime << "; factor time = "
			<< factor_basewtime << ", apply time = "<< apply_basewtime << "\n#---\n#";
		outf << std::setw(w) << "threads"
			<< std::setw(w) << "b&a-sweeps"
			<< std::setw(w+5) << "factor-speedup" << std::setw(w+5) << "apply-speedup" << std::setw(w+5)
			<< "total-speedup"
			<< std::setw(w) << "cpu-time" 
			<< std::setw(w+6) << "total-lin-iters" << std::setw(w+5) << "avg-lin-iters"
			<< std::setw(w+1) << "time-steps"<< std::setw(w) << "converged?" << "\n#---\n";

		outf << "#" <<std::setw(w) << 1 
			<< std::setw(w/2) << 1 << std::setw(w/2) << 1 
			<< std::setw(w+5) << 1.0 << std::setw(w+5) << 1.0 << std::setw(w+5) << 1.0
			<< std::setw(w) << bctx.factorcputime + bctx.applycputime
			<< std::setw(w+6) << tdata.total_lin_iters
			<< std::setw(w+5) << tdata.avg_lin_iters << std::setw(w+1) << tdata.num_timesteps 
			<< std::setw(w) << (tdata.converged ? 1 : 0) << "\n#---\n" << std::flush;
	}

	omp_set_num_threads(numthreads);
	
	// Carry out multi-thread run
	// Note: The run is regarded as converged only if each repetition converged
	for (const int nswp : sweep_seq)
	{
		TimingData tdata = {0,0,0,0,0,0,0,0,0,true};
		double precwalltime = 0, preccputime = 0, factorwalltime = 0, applywalltime = 0;
		const int naswp = std::round(sweepratio*nswp);

		int irpt;
		for(irpt = 0; irpt < numrepeat; irpt++) 
		{
			TimingData td = run_sweeps(startprob, prob, maintconf, nswp, naswp, 
					&ksp, u, A, M, mfjac, mf_flg, bctx);
			
			computeTotalTimes(&bctx);

			tdata.nelem = td.nelem;
			tdata.num_threads = td.num_threads;
			tdata.lin_walltime += td.lin_walltime;
			tdata.lin_cputime +=  td.lin_cputime;
			tdata.ode_walltime += td.ode_walltime;
			tdata.ode_cputime +=  td.ode_cputime;
			tdata.total_lin_iters += td.total_lin_iters;
			tdata.num_timesteps += td.num_timesteps;
			tdata.converged = tdata.converged && td.converged;
			factorwalltime += bctx.factorwalltime;
			applywalltime += bctx.applywalltime;
			precwalltime += bctx.factorwalltime + bctx.applywalltime;
			preccputime += bctx.factorcputime + bctx.applycputime;

			if(!td.converged) {
				irpt++;
				break;
			}
		}

		tdata.lin_walltime /= (double)irpt;
		tdata.lin_cputime /= (double)irpt;
		tdata.ode_walltime /= (double)irpt;
		tdata.ode_cputime /= (double)irpt;
		tdata.num_timesteps /= (double)irpt;
		tdata.total_lin_iters /= (double)irpt;
		tdata.avg_lin_iters = tdata.total_lin_iters / (double)tdata.num_timesteps;
		factorwalltime /= (double)irpt;
		applywalltime /= (double)irpt;
		precwalltime /= (double)irpt;
		preccputime /= (double)irpt;

		if(mpirank == 0) {
			outf << ' ' << std::setw(w) << numthreads 
				<< std::setw(w/2) << nswp << std::setw(w/2) << naswp
				<< std::setw(w+5) << factor_basewtime/factorwalltime
				<< std::setw(w+5) << apply_basewtime/applywalltime
				<< std::setw(w+5) << prec_basewtime/precwalltime
				<< std::setw(w) << preccputime
				<< std::setw(w+6) << tdata.total_lin_iters
				<< std::setw(w+5) << tdata.avg_lin_iters << std::setw(w+1) << tdata.num_timesteps 
				<< std::setw(w) << (tdata.converged ? 1:0) << '\n' << std::flush;
		}
	}

	delete time;
	delete starttime;
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	destroyBlastedDataList(&bctx);
	ierr = MatDestroy(&M); CHKERRQ(ierr);
	if(mf_flg) {
		ierr = MatDestroy(&A); 
		CHKERRQ(ierr);
	}
	delete prob;
	delete startprob;
	
	return ierr;
}

TimingData run_sweeps(const Spatial<NVARS> *const startprob, const Spatial<NVARS> *const prob,
		const SteadySolverConfig& maintconf, const int nbswps, const int naswps,
		KSP *ksp, Vec u, Mat A, Mat M, MatrixFreeSpatialJacobian<NVARS>& mfjac, const PetscBool mf_flg,
		Blasted_data_list& bctx)
{
	StatusCode ierr = 0;

	set_blasted_sweeps(nbswps,naswps);

	std::cout << "Using sweeps " << nbswps << "," << naswps << ".\n";

	// Reset the KSP
	ierr = KSPDestroy(ksp); petsc_throw(ierr, "run_sweeps: Couldn't destroy KSP");
	destroyBlastedDataList(&bctx);
	ierr = KSPCreate(PETSC_COMM_WORLD, ksp); petsc_throw(ierr, "run_sweeps: Couldn't create KSP");
	if(mf_flg) {
		ierr = KSPSetOperators(*ksp, A, M); 
		petsc_throw(ierr, "run_sweeps: Couldn't set KSP operators");
	}
	else {
		ierr = KSPSetOperators(*ksp, M, M); 
		petsc_throw(ierr, "run_sweeps: Couldn't set KSP operators");
	}
	ierr = KSPSetFromOptions(*ksp); petsc_throw(ierr, "run_sweeps: Couldn't set KSP from options");
	
	bctx = newBlastedDataList();
	ierr = setup_blasted<NVARS>(*ksp,u,startprob,bctx);
	fvens_throw(ierr, "run_sweeps: Couldn't setup BLASTed");

	// setup nonlinear ODE solver for main solve
	SteadyBackwardEulerSolver<NVARS>* time 
		= new SteadyBackwardEulerSolver<NVARS>(prob, maintconf, *ksp);
	std::cout << " Set up backward Euler temporal scheme for main solve.\n";

	mfjac.set_spatial(prob);

	Vec ut;
	ierr = MatCreateVecs(M, &ut, NULL); petsc_throw(ierr, "Couldn't create vec");
	ierr = VecCopy(u, ut); petsc_throw(ierr, "run_sweeps: Couldn't copy vec");
	
	ierr = time->solve(ut); fvens_throw(ierr, "run_sweeps: Couldn't solve ODE");
	const TimingData tdata = time->getTimingData();

	delete time;
	ierr = VecDestroy(&ut); petsc_throw(ierr, "run_sweeps: Couldn't delete vec");

	return tdata;
}

void set_blasted_sweeps(const int nbswp, const int naswp)
{
	// add option
	std::string value = std::to_string(nbswp) + "," + std::to_string(naswp);
	int ierr = PetscOptionsSetValue(NULL, "-blasted_async_sweeps", value.c_str());
	petsc_throw(ierr, "Couldn't set PETSc option for BLASTed async sweeps");

	// Check
	int checksweeps[2];
	int nmax = 2;
	PetscBool set = PETSC_FALSE;
	ierr = PetscOptionsGetIntArray(NULL,NULL,"-blasted_async_sweeps",checksweeps,&nmax,&set);
	petsc_throw(ierr, "Could not get int array!");
	fvens_throw(checksweeps[0] != nbswp || checksweeps[1] != naswp, 
			"Async sweeps not set properly!");
}

}


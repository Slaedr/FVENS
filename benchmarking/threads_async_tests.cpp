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
#include "utilities/aerrorhandling.hpp"
#include "linalg/alinalg.hpp"
#include "ode/aodesolver.hpp"
#include "mesh/ameshutils.hpp"

#include <blasted_petsc.h>

#include "threads_async_tests.hpp"

namespace benchmark {

using namespace fvens;

/// Set -blasted_async_sweeps in the default Petsc options database and throw if not successful
static void set_blasted_sweeps(const int nbswp, const int naswp);

static void writeHeaderToFile(std::ofstream& outf, const int width)
{
	outf << '#' << std::setw(width) << "threads"
	     << std::setw(width) << "b&a-sweeps"
	     << std::setw(width+5) << "factor-speedup" << std::setw(width+5) << "apply-speedup"
	     << std::setw(width+5) << "total-speedup" << std::setw(width+5) << "total-deviate"
	     << std::setw(width) << "cpu-time" << std::setw(width+6) << "total-lin-iters"
	     << std::setw(width+4) << "avg-lin-iters"
	     << std::setw(width+1) << "time-steps"<< std::setw(width/2+1) << "conv?"
	     << std::setw(width) << "nl-speedup"
	     << "\n#---\n";
}

static void writeTimingToFile(std::ofstream& outf, const int w, const bool comment,
                              const TimingData& tdata, const int numthreads,
                              const int nbswps, const int naswps,
                              const double factorspeedup, const double applyspeedup,
                              const double precspeedup, const double precdeviate,
                              const double preccputime, const double fvens_wall_spdp)
{
	outf << (comment ? '#' : ' ') << std::setw(w) << numthreads
	     << std::setw(w/2) << nbswps << std::setw(w/2) << naswps
	     << std::setw(w+5) << factorspeedup << std::setw(w+5) << applyspeedup
	     << std::setw(w+5) << precspeedup << std::setw(w+5) << precdeviate
	     << std::setw(w) << preccputime
	     << std::setw(w+6) << tdata.total_lin_iters << std::setw(w+4) << tdata.avg_lin_iters
	     << std::setw(w+1) << tdata.num_timesteps
	     << std::setw(w/2+1) << (tdata.converged ? 1 : 0)
	     << std::setw(w) << fvens_wall_spdp
	     << (comment ? "\n#---\n" : "\n") << std::flush;
}

static double std_deviation(const double *const vals, const double avg, const int N) {
	double deviate = 0;
	for(int j = 0; j < N; j++)
		deviate += (vals[j]-avg)*(vals[j]-avg);
	deviate = std::sqrt(deviate/(double)N);
	return deviate;
}

StatusCode test_speedup_sweeps(const FlowParserOptions& opts, const int numrepeat,
                               const std::vector<int>& threads_seq,
                               const std::vector<int>& bswpseq, const std::vector<int>& aswpseq,
                               std::ofstream& outf)
{
	StatusCode ierr = 0;

	// Set up mesh
	UMesh2dh<a_real> m;
	m.readMesh(opts.meshfile);
	CHKERRQ(preprocessMesh(m));
	// check if there are any periodic boundaries
	//m.compute_periodic_map(opts.periodic_marker, opts.periodic_axis);
	for(auto it = opts.bcconf.begin(); it != opts.bcconf.end(); it++) {
		if(it->bc_type == PERIODIC_BC)
			m.compute_periodic_map(it->bc_opts[0], it->bc_opts[1]);
	}

	std::cout << "\n***\n";

	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// numerics for main solver
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);
	// simpler numerics for startup
	const FlowNumericsConfig nconfstart {opts.invflux, opts.invfluxjac, "NONE", "NONE", false};

	std::cout << "Setting up main spatial scheme.\n";
	const Spatial<a_real,NVARS> *const prob = create_const_flowSpatialDiscretization(&m, pconf, nconfmain);
	std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<a_real,NVARS> *const startprob
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

	SteadySolverConfig maintconf {
		opts.lognres, opts.logfile,
		opts.initcfl, opts.endcfl, opts.rampstart, opts.rampend,
		opts.tolerance, opts.maxiter,
	};

	const SteadySolverConfig starttconf {
		opts.lognres, opts.logfile+"-init",
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

	// change file to which residual history is written
	maintconf.logfile = opts.logfile + "-sweeps11-serial";

	omp_set_num_threads(1);

	TimingData tdata = run_sweeps(startprob, prob, maintconf, 1, 1, &ksp, u, A, M, 
			mfjac, mf_flg, bctx);

	computeTotalTimes(&bctx);
	const double factor_basewtime = bctx.factorwalltime;
	const double apply_basewtime = bctx.applywalltime;
	const double prec_basewtime = bctx.factorwalltime + bctx.applywalltime;
	const double ode_basewtime = tdata.ode_walltime;

	// Field width in the report file
	const int w = 11;

	if(mpirank == 0) {
		outf << "# Base preconditioner wall time = " << prec_basewtime << "; factor time = "
		     << factor_basewtime << ", apply time = "<< apply_basewtime
		     << ", nonlinear solve time = " << ode_basewtime << "\n#---\n";

		writeHeaderToFile(outf,w);
		writeTimingToFile(outf, w, true,tdata, 1, 1,1, 1.0, 1.0, 1.0,
		                  0.0, bctx.factorcputime+bctx.applycputime, 1.0);
	}

	for(int numthreads : threads_seq)
	{
		omp_set_num_threads(numthreads);

		// Carry out multi-thread run
		// Note: The run is regarded as converged only if each repetition converged
		for (size_t i = 0; i < bswpseq.size(); i++)
		{
			TimingData tdata = {0,0,0,0,0,0,0,0,0,true};
			double precwalltime = 0, preccputime = 0, factorwalltime = 0, applywalltime = 0;
			std::vector<double> precwalltimearr(numrepeat);

			int irpt;
			for(irpt = 0; irpt < numrepeat; irpt++) 
			{
				// change file to which residual history is written
				// Only write residual history for last run
				if(irpt != numrepeat-1)
					maintconf.lognres = false;
				else {
					maintconf.lognres = opts.lognres;
					maintconf.logfile = opts.logfile + "-sweeps"
						+std::to_string(bswpseq[i])+std::to_string(aswpseq[i])+"-"
						+std::to_string(numthreads)+"threads";
				}

				TimingData td = run_sweeps(startprob, prob, maintconf, bswpseq[i], aswpseq[i], 
						&ksp, u, A, M, mfjac, mf_flg, bctx);

				computeTotalTimes(&bctx);

				if(!td.converged) {
					tdata.converged = false;
					break;
				}

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
				precwalltimearr[irpt] = bctx.factorwalltime + bctx.applywalltime;
			}

			tdata.lin_walltime /= (double)irpt;
			tdata.lin_cputime /= (double)irpt;
			tdata.ode_walltime /= (double)irpt;
			tdata.ode_cputime /= (double)irpt;
			tdata.num_timesteps = (int) std::round(tdata.num_timesteps / (double)irpt);
			tdata.total_lin_iters = (int) std::round(tdata.total_lin_iters / (double)irpt);
			tdata.avg_lin_iters = (int)std::round(tdata.total_lin_iters/(double)tdata.num_timesteps);
			factorwalltime /= (double)irpt;
			applywalltime /= (double)irpt;
			precwalltime /= (double)irpt;
			preccputime /= (double)irpt;

			const double precdeviate = std_deviation(&precwalltimearr[0], precwalltime, irpt);

			if(mpirank == 0) {
				writeTimingToFile(outf, w, false,tdata, numthreads, bswpseq[i], aswpseq[i],
				                  factor_basewtime/factorwalltime,apply_basewtime/applywalltime,
				                  prec_basewtime/precwalltime, precdeviate, preccputime,
				                  ode_basewtime/tdata.ode_walltime);
			}
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

TimingData run_sweeps(const Spatial<a_real,NVARS> *const startprob, const Spatial<a_real,NVARS> *const prob,
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

	try {
		ierr = time->solve(ut);
	}
	catch (Numerical_error& e) {
		std::cout << "There was a numerical error in the steady state solve!\n";
		TimingData tdata; tdata.converged = false;
		delete time;
		ierr = VecDestroy(&ut); petsc_throw(ierr, "run_sweeps: Couldn't delete vec");
		return tdata;
	}

	fvens_throw(ierr, "run_sweeps: Couldn't solve ODE");
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


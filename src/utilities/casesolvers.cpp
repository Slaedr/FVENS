/** \file casesolvers.cpp
 * \brief Routines to solve a single fluid dynamics case
 * \author Aditya Kashi
 * \date 2018-04
 */

#include <iostream>
#include <iomanip>
#include <tuple>

#include "casesolvers.hpp"
#include "utilities/afactory.hpp"
#include "utilities/aoptionparser.hpp"
#include "spatial/aoutput.hpp"
#include "mesh/ameshutils.hpp"
#include "mpiutils.hpp"

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

namespace fvens {

/// Prepare a mesh for use in a fluid simulation
UMesh2dh<a_real> constructMeshFlow(const FlowParserOptions& opts, const std::string mesh_suffix)
{
	// Set up mesh
	const std::string meshfile = opts.meshfile + mesh_suffix;
	UMesh2dh<a_real> m(constructMesh(meshfile));

	// check if there are any periodic boundaries
	for(auto it = opts.bcconf.begin(); it != opts.bcconf.end(); it++) {
		if(it->bc_type == PERIODIC_BC)
			m.compute_periodic_map(it->bc_opts[0], it->bc_opts[1]);
	}

	return m;
}

const FlowFV_base<a_real>* createFlowSpatial(const FlowParserOptions& opts,
                                             const UMesh2dh<a_real>& m)
{
	std::cout << "Setting up main spatial scheme.\n";
	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// numerics for main solver
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);

	return create_const_flowSpatialDiscretization(&m, pconf, nconfmain);
}

int initializeSystemVector(const FlowParserOptions& opts, const UMesh2dh<a_real>& m, Vec *const u)
{
	//int ierr = VecCreateSeq(PETSC_COMM_SELF, m.gnelem()*NVARS, u); CHKERRQ(ierr);
	int ierr = createGhostedSystemVector(&m, NVARS, u); CHKERRQ(ierr);

	const IdealGasPhysics<a_real> phy(opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr);
	const std::array<a_real,NVARS> uinf = phy.compute_freestream_state(opts.alpha);

	MutableGhostedVecHandler<PetscScalar> uh(*u);
	PetscScalar *const uloc = uh.getArray();
	
	//initial values are equal to free-stream values
	for(a_int i = 0; i < m.gnelem()+m.gnConnFace(); i++)
		for(int j = 0; j < NVARS; j++)
			uloc[i*NVARS+j] = uinf[j];

	return ierr;
}

FlowCase::FlowCase(const FlowParserOptions& options) : opts{options}
{
}

int FlowCase::run(const UMesh2dh<a_real>& m, Vec u) const
{
	int ierr = 0;
	const Spatial<a_real,NVARS> *const prob = createFlowSpatial(opts, m);

	ierr = execute(prob, false, u); CHKERRQ(ierr);

	delete prob;
	return ierr;
}

FlowSolutionFunctionals FlowCase::run_output(const bool surface_file_needed,
											 const bool vtu_output_needed,
                                             const UMesh2dh<a_real>& m, Vec u) const
{
	int ierr = 0;
	std::cout << "\nSetting up flow case with output\n";
	const int mpisize = get_mpi_size(PETSC_COMM_WORLD);
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	const FlowFV_base<a_real> *const prob = createFlowSpatial(opts, m);

	const a_real h = 1.0 / ( std::pow((a_real)m.gnelem(), 1.0/NDIM) );

	try {
		ierr = execute(prob, opts.lognres, u);
	}
	catch (Tolerance_error& e) {
		std::cout << e.what() << std::endl;
	}
	fvens_throw(ierr, "Could not solve steady case! Error code " + std::to_string(ierr));

	MVector<a_real> umat; umat.resize(m.gnelem(),NVARS);
	const PetscScalar *uarr;
	ierr = VecGetArrayRead(u, &uarr); 
	petsc_throw(ierr, "Petsc VecGetArrayRead error");
	for(a_int i = 0; i < m.gnelem(); i++)
		for(int j = 0; j < NVARS; j++)
			umat(i,j) = uarr[i*NVARS+j];
	ierr = VecRestoreArrayRead(u, &uarr); 
	petsc_throw(ierr, "Petsc VecRestoreArrayRead error");

	IdealGasPhysics<a_real> phy(opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr);
	FlowOutput out(prob, &phy, opts.alpha);

	const a_real entropy = out.compute_entropy_cell(u);

	// Currently, ouput to files only works in single-process runs
	if(mpisize == 1) {
		if(surface_file_needed) {
			try {
				out.exportSurfaceData(umat, opts.lwalls, opts.lothers, opts.surfnameprefix);
			} 
			catch(std::exception& e) {
				std::cout << e.what() << std::endl;
			}
		}
	
		if(vtu_output_needed) {
			amat::Array2d<a_real> scalars;
			amat::Array2d<a_real> velocities;
			out.postprocess_point(u, scalars, velocities);

			std::string scalarnames[] = {"density", "mach-number", "pressure", "temperature"};
			writeScalarsVectorToVtu_PointData(opts.vtu_output_file,
			                                  m, scalars, scalarnames, velocities, "velocity");
		}

		if(opts.vol_output_reqd == "YES")
			out.exportVolumeData(umat, opts.volnameprefix);
	}
	else {
		if(mpirank == 0)
			std::cout << "FlowCase: Files will not be written in multi-process runs.\n";
	}
	
	MVector<a_real> output; output.resize(m.gnelem(),NDIM+2);
	std::vector<GradBlock_t<a_real,NDIM,NVARS>> grad;
	grad.resize(m.gnelem());
	prob->getGradients(umat, &grad[0]);

	const std::tuple<a_real,a_real,a_real> fnls 
		{ prob->computeSurfaceData(umat, &grad[0], opts.lwalls[0], output)};

	delete prob;

	return FlowSolutionFunctionals{h, entropy,
			std::get<0>(fnls), std::get<1>(fnls), std::get<2>(fnls)};
}

void FlowCase::setupKSP(LinearProblemLHS& solver, const bool use_mfjac) {
	// initialize solver
	int ierr = KSPCreate(PETSC_COMM_WORLD, &solver.ksp); petsc_throw(ierr, "KSP Create");
	if(use_mfjac) {
		ierr = KSPSetOperators(solver.ksp, solver.A, solver.M); 
		petsc_throw(ierr, "KSP set operators");
	}
	else {
		ierr = KSPSetOperators(solver.ksp, solver.M, solver.M); 
		petsc_throw(ierr, "KSP set operators");
	}

	ierr = KSPSetFromOptions(solver.ksp); petsc_throw(ierr, "KSP set from options");
}

FlowCase::LinearProblemLHS FlowCase::setupImplicitSolver(const Spatial<a_real,NVARS> *const space,
                                                         const bool use_mfjac)
{
	LinearProblemLHS solver;
	const UMesh2dh<a_real> *const mesh = space->mesh();

	// Initialize Jacobian for implicit schemes
	int ierr = setupSystemMatrix<NVARS>(mesh, &solver.M); fvens_throw(ierr, "Setup system matrix");

	// setup matrix-free Jacobian if requested
	if(use_mfjac) {
		std::cout << " Allocating matrix-free Jac\n";
		ierr = create_matrixfree_jacobian<NVARS>(space, &solver.A); 
		fvens_throw(ierr, "Setup matrix-free Jacobian");
	}

	setupKSP(solver, use_mfjac);
	solver.mf_flg = use_mfjac;

	return solver;
}

SteadyFlowCase::SteadyFlowCase(const FlowParserOptions& options)
	: FlowCase(options),
	  mf_flg {parsePetscCmd_isDefined("-matrix_free_jacobian")}
{ }

int SteadyFlowCase::execute_starter(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);
	
	const UMesh2dh<a_real> *const m = prob->mesh();

	const FlowPhysicsConfig pconf {extract_spatial_physics_config(opts)};
	const FlowNumericsConfig nconfstart {firstorder_spatial_numerics_config(opts)};

	if(mpirank == 0)
		std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<a_real,NVARS> *const startprob
		= create_const_flowSpatialDiscretization(m, pconf, nconfstart);

	if(mpirank == 0)
		std::cout << "***\n";

	LinearProblemLHS isol = setupImplicitSolver(startprob, mf_flg);

	// set up time discrization

	const SteadySolverConfig starttconf {
		opts.lognres, opts.logfile+"-init",
		opts.firstinitcfl, opts.firstendcfl, opts.firstrampstart, opts.firstrampend,
		opts.firsttolerance, opts.firstmaxiter,
	};

	SteadySolver<NVARS> * starttime = nullptr;
	const NonlinearUpdate<NDIM+2> *nlupdate = nullptr;

	if(opts.pseudotimetype == "IMPLICIT")
	{
		nlupdate = create_const_nonlinearUpdateScheme<NDIM+2>(opts);
		if(opts.usestarter != 0) {
			starttime = new SteadyBackwardEulerSolver<NVARS>(startprob, starttconf, isol.ksp,
			                                                 nlupdate);
			if(mpirank == 0)
				std::cout << "Set up backward Euler temporal scheme for initialization solve.\n";
		}

	}
	else
	{
		if(opts.usestarter != 0) {
			starttime = new SteadyForwardEulerSolver<NVARS>(startprob, u, starttconf);
			if(mpirank == 0)
				std::cout << "Set up explicit forward Euler temporal scheme for startup solve.\n";
		}
	}

	// setup BLASTed preconditioning if requested
#ifdef USE_BLASTED
	Blasted_data_list bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(isol.ksp,u,startprob,bctx); CHKERRQ(ierr);
	}
#endif

	if(mpirank == 0)
		std::cout << "***\n";

	// computation

	if(opts.usestarter != 0)
	{
		// Solve the starter problem to get the initial solution
		// If the starting solve does not converge to the required tolerance, don't throw and
		// move on.
		try {
			ierr = starttime->solve(u); CHKERRQ(ierr);
		} catch (Tolerance_error& e) {
			std::cout << e.what() << std::endl;
		}
	}

	delete starttime;
	delete nlupdate;

	// Note that we destroy the KSP after the startup solve (in fact, after any solve)
	//  - could be advantageous for some types of algebraic solvers
	//  This will also reset the BLASTed timing counters.
	ierr = isol.destroy(); CHKERRQ(ierr);
#ifdef USE_BLASTED
	destroyBlastedDataList(&bctx);
#endif

	delete startprob;
	return ierr;
}

TimingData SteadyFlowCase::execute_main(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	LinearProblemLHS isol = setupImplicitSolver(prob, mf_flg);

	// set up time discrization

	const SteadySolverConfig maintconf {
		opts.lognres, opts.logfile,
		opts.initcfl, opts.endcfl, opts.rampstart, opts.rampend,
		opts.tolerance, opts.maxiter,
	};

	SteadySolver<NDIM+2> * time = nullptr;
	const NonlinearUpdate<NDIM+2> *nlupdate = nullptr;

#ifdef USE_BLASTED
	Blasted_data_list bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(isol.ksp,u,prob,bctx); fvens_throw(ierr, "BLASTed not setup");
	}
#endif

	// setup nonlinear ODE solver for main solve - MUST be done AFTER KSPCreate
	if(opts.pseudotimetype == "IMPLICIT")
	{
		nlupdate = create_const_nonlinearUpdateScheme<NDIM+2>(opts);
		time = new SteadyBackwardEulerSolver<NVARS>(prob, maintconf, isol.ksp, nlupdate);
		if(mpirank == 0)
			std::cout << "\nSet up backward Euler temporal scheme for main solve.\n";
	}
	else
	{
		time = new SteadyForwardEulerSolver<NVARS>(prob, u, maintconf);
		if(mpirank == 0)
			std::cout << "\nSet up explicit forward Euler temporal scheme for main solve.\n";
	}

	// Solve the main problem
	try {
		ierr = time->solve(u);
	}
	catch(Numerical_error& e) {
		std::cout << "FVENS: Main solve failed: " << e.what() << std::endl;
		TimingData tdata; tdata.converged = false;
		delete time;
		return tdata;
	}

	petsc_throw(ierr, "Nonlinear solver failed!");
	if(mpirank == 0)
		std::cout << "***\n";

	TimingData tdata = time->getTimingData();
#ifdef USE_BLASTED
	computeTotalTimes(&bctx);
	tdata.precsetup_walltime = bctx.factorwalltime;
	tdata.precapply_walltime = bctx.applywalltime;
	tdata.prec_cputime = bctx.factorcputime + bctx.applycputime;
#endif

	delete time;
	delete nlupdate;
	ierr = isol.destroy(); petsc_throw(ierr, "Could not destroy linear problen LHS");
#ifdef USE_BLASTED
	destroyBlastedDataList(&bctx);
#endif

	return tdata;
}

int SteadyFlowCase::execute(const Spatial<a_real,NVARS> *const prob, const bool outhist, Vec u) const
{
	int ierr = 0;
	
	ierr = execute_starter(prob, u); fvens_throw(ierr, "Startup solve failed!");
	TimingData td = execute_main(prob, u); fvens_throw(ierr, "Steady case solver failed!");
	if(!td.converged)
		throw Tolerance_error("Main flow solve did not converge!");

	if(outhist) {
		const int rank = get_mpi_rank(PETSC_COMM_WORLD);
		//MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		if(rank == 0) {
			std::ofstream convout(opts.logfile+"-residual_history.log");
			writeConvergenceHistoryHeader(convout);
			for(int istp = 0; istp < td.num_timesteps; istp++)
				writeStepToConvergenceHistory(td.convhis[istp], convout);
			convout.close();

			convout.open(opts.logfile+"-prec-timing.log");
			convout << std::setw(10) << "# N.threads" << std::setw(18) << "Prec.setup wtime"
			        << std::setw(18) << "Prec.apply wtime" << std::setw(10) << "Prec.CPU"
			        << std::setw(15) << "Tot.lin.iters" << std::setw(15) << "Avg.lin.iters"
			        << std::setw(12) << "Time-steps" << std::setw(12) << "converged?" << '\n';
			convout << std::setw(10) << td.num_threads << std::setw(18) << td.precsetup_walltime
			        << std::setw(18) << td.precapply_walltime << std::setw(10) << td.prec_cputime
			        << std::setw(15) << td.total_lin_iters << std::setw(15) << td.avg_lin_iters
			        << std::setw(12) << td.num_timesteps << std::setw(12) << (td.converged ? 1:0)
			        << '\n';
			convout.close();
		}
	}

	return ierr;
}

UnsteadyFlowCase::UnsteadyFlowCase(const FlowParserOptions& options)
	: FlowCase(options)
{ }

/** \todo Implement an unsteady integrator factory and use that here.
 */
int UnsteadyFlowCase::execute(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;

	if(opts.time_integrator == "TVDRK") {
		TVDRKSolver<NVARS> time(prob, u, opts.time_order, opts.logfile, opts.phy_cfl);
		ierr = time.solve(opts.final_time);
		CHKERRQ(ierr);
		return ierr;
	} else {
		throw "Nothing but TVDRK is implemented yet!";
	}

	return ierr;
}

}

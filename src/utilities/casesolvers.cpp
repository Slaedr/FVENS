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
#include "utilities/aerrorhandling.hpp"
#include "utilities/aoptionparser.hpp"
#include "spatial/aoutput.hpp"
#include "mesh/ameshutils.hpp"

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

namespace fvens {

FlowCase::FlowCase(const FlowParserOptions& options) : opts{options}
{
}

/// Prepare a mesh for use in a fluid simulation
UMesh2dh<a_real> FlowCase::constructMesh(const std::string mesh_suffix) const
{
	// Set up mesh
	const std::string meshfile = opts.meshfile + mesh_suffix;
	UMesh2dh<a_real> m;
	m.readMesh(meshfile);
	int ierr = preprocessMesh(m); 
	fvens_throw(ierr, "Mesh could not be preprocessed!");

	// check if there are any periodic boundaries
	for(auto it = opts.bcconf.begin(); it != opts.bcconf.end(); it++) {
		if(it->bc_type == PERIODIC_BC)
			m.compute_periodic_map(it->bc_opts[0], it->bc_opts[1]);
	}

	return m;
}

int FlowCase::run(const std::string mesh_suffix, Vec *const u) const
{
	int ierr = 0;
	
	const UMesh2dh<a_real> m = constructMesh(mesh_suffix);
	std::cout << "***\n";

	std::cout << "Setting up main spatial scheme.\n";
	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// numerics for main solver
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);
	const Spatial<a_real,NVARS> *const prob = create_const_flowSpatialDiscretization(&m, pconf, nconfmain);

	ierr = VecCreateSeq(PETSC_COMM_SELF, m.gnelem()*NVARS, u);
	prob->initializeUnknowns(*u);

	ierr = execute(prob, *u); CHKERRQ(ierr);

	delete prob;

	return ierr;
}

FlowSolutionFunctionals FlowCase::run_output(const std::string mesh_suffix, 
											 const bool vtu_output_needed, Vec *const u) const
{
	int ierr = 0;
	
	const UMesh2dh<a_real> m = constructMesh(mesh_suffix);
	const a_real h = 1.0 / ( std::pow((a_real)m.gnelem(), 1.0/NDIM) );
	std::cout << "***\n";

	std::cout << "Setting up main spatial scheme.\n";
	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// numerics for main solver
	const FlowNumericsConfig nconfmain = extract_spatial_numerics_config(opts);
	const FlowFV_base<a_real> *const prob
		= create_const_flowSpatialDiscretization(&m, pconf, nconfmain);

	ierr = VecCreateSeq(PETSC_COMM_SELF, m.gnelem()*NVARS, u);
	prob->initializeUnknowns(*u);

	try {
		ierr = execute(prob, *u);
	}
	catch (Tolerance_error& e) {
		std::cout << e.what() << std::endl;
	}
	fvens_throw(ierr, "Could not solve steady case! Error code " + std::to_string(ierr));

	MVector<a_real> umat; umat.resize(m.gnelem(),NVARS);
	const PetscScalar *uarr;
	ierr = VecGetArrayRead(*u, &uarr); 
	petsc_throw(ierr, "Petsc VecGetArrayRead error");
	for(a_int i = 0; i < m.gnelem(); i++)
		for(int j = 0; j < NVARS; j++)
			umat(i,j) = uarr[i*NVARS+j];
	ierr = VecRestoreArrayRead(*u, &uarr); 
	petsc_throw(ierr, "Petsc VecRestoreArrayRead error");

	IdealGasPhysics<a_real> phy(opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr);
	FlowOutput out(prob, &phy, opts.alpha);

	const a_real entropy = out.compute_entropy_cell(*u);

	try {
		out.exportSurfaceData(umat, opts.lwalls, opts.lothers, opts.surfnameprefix);
	} 
	catch(std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	
	if(vtu_output_needed) {
		amat::Array2d<a_real> scalars;
		amat::Array2d<a_real> velocities;
		out.postprocess_point(*u, scalars, velocities);

		std::string scalarnames[] = {"density", "mach-number", "pressure", "temperature"};
		writeScalarsVectorToVtu_PointData(opts.vtu_output_file,
		                                  m, scalars, scalarnames, velocities, "velocity");
	}
	
	MVector<a_real> output; output.resize(m.gnelem(),NDIM+2);
	GradArray<a_real,NVARS> grad;
	grad.resize(m.gnelem());
	prob->getGradients(umat, grad);
	const std::tuple<a_real,a_real,a_real> fnls 
		{ prob->computeSurfaceData(umat, grad, opts.lwalls[0], output)};

	if(opts.vol_output_reqd == "YES")
		out.exportVolumeData(umat, opts.volnameprefix);

	delete prob;

	return FlowSolutionFunctionals{h, entropy,
			std::get<0>(fnls), std::get<1>(fnls), std::get<2>(fnls)};
}

void FlowCase::setupKSP(ImplicitSolver& solver, const bool use_mfjac) {
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

FlowCase::ImplicitSolver FlowCase::setupImplicitSolver(const UMesh2dh<a_real> *const mesh,
                                                       const bool use_mfjac)
{
	ImplicitSolver solver;

	// Initialize Jacobian for implicit schemes
	int ierr = setupSystemMatrix<NVARS>(mesh, &solver.M); fvens_throw(ierr, "Setup system matrix");

	// setup matrix-free Jacobian if requested
	if(use_mfjac) {
		std::cout << " Allocating matrix-free Jac\n";
		ierr = setup_matrixfree_jacobian<NVARS>(mesh, &solver.mfjac, &solver.A); 
		fvens_throw(ierr, "Setup matrix-free Jacobian");
	}

	setupKSP(solver, use_mfjac);

	return solver;
}

SteadyFlowCase::SteadyFlowCase(const FlowParserOptions& options)
	: FlowCase(options),
	  pconf {extract_spatial_physics_config(opts)},
	  nconfstart {firstorder_spatial_numerics_config(opts)},
	  mf_flg {parsePetscCmd_isDefined("-matrix_free_jacobian")}
{ }

int SteadyFlowCase::execute_starter(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;
	
	const UMesh2dh<a_real> *const m = prob->mesh();

	std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<a_real,NVARS> *const startprob
		= create_const_flowSpatialDiscretization(m, pconf, nconfstart);

	std::cout << "***\n";

	ImplicitSolver isol = setupImplicitSolver(m, mf_flg);

	// set up time discrization

	const SteadySolverConfig starttconf {
		opts.lognres, opts.logfile+"-init",
		opts.firstinitcfl, opts.firstendcfl, opts.firstrampstart, opts.firstrampend,
		opts.firsttolerance, opts.firstmaxiter,
	};

	SteadySolver<NVARS> * starttime = nullptr;

	if(opts.pseudotimetype == "IMPLICIT")
	{
		if(opts.usestarter != 0) {
			starttime = new SteadyBackwardEulerSolver<NVARS>(startprob, starttconf, isol.ksp);
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

	// setup BLASTed preconditioning if requested
#ifdef USE_BLASTED
	Blasted_data_list bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(isol.ksp,u,startprob,bctx); CHKERRQ(ierr);
	}
#endif

	std::cout << "***\n";

	// computation

	if(opts.usestarter != 0) {

		isol.mfjac.set_spatial(startprob);

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

	// Reset the KSP - could be advantageous for some types of algebraic solvers
	//  This will also reset the BLASTed timing counters.
	ierr = KSPDestroy(&isol.ksp); CHKERRQ(ierr);
#ifdef USE_BLASTED
	destroyBlastedDataList(&bctx);
#endif
	ierr = MatDestroy(&isol.M); CHKERRQ(ierr);
	if(mf_flg) {
		ierr = MatDestroy(&isol.A); 
		CHKERRQ(ierr);
	}

	delete startprob;
	return ierr;
}

TimingData SteadyFlowCase::execute_main(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;
	
	const UMesh2dh<a_real> *const m = prob->mesh();

	ImplicitSolver isol = setupImplicitSolver(m, mf_flg);

	// set up time discrization

	const SteadySolverConfig maintconf {
		opts.lognres, opts.logfile,
		opts.initcfl, opts.endcfl, opts.rampstart, opts.rampend,
		opts.tolerance, opts.maxiter,
	};

	SteadySolver<NVARS> * time = nullptr;

#ifdef USE_BLASTED
	Blasted_data_list bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(isol.ksp,u,prob,bctx); fvens_throw(ierr, "BLASTed not setup");
	}
#endif

	// setup nonlinear ODE solver for main solve - MUST be done AFTER KSPCreate
	if(opts.pseudotimetype == "IMPLICIT")
	{
		time = new SteadyBackwardEulerSolver<NVARS>(prob, maintconf, isol.ksp);
		std::cout << "\nSet up backward Euler temporal scheme for main solve.\n";
	}
	else
	{
		time = new SteadyForwardEulerSolver<NVARS>(prob, u, maintconf);
		std::cout << "\nSet up explicit forward Euler temporal scheme for main solve.\n";
	}

	isol.mfjac.set_spatial(prob);

	// Solve the main problem
	try {
		ierr = time->solve(u);
	}
	catch(Numerical_error& e) {
		TimingData tdata; tdata.converged = false;
		delete time;
		return tdata;
	}

	fvens_throw(ierr, "Nonlinear solver failed!");
	std::cout << "***\n";

	const TimingData tdata = time->getTimingData();

	delete time;
	ierr = KSPDestroy(&isol.ksp); petsc_throw(ierr, "KSP destroy");
#ifdef USE_BLASTED
	destroyBlastedDataList(&bctx);
#endif
	ierr = MatDestroy(&isol.M); petsc_throw(ierr, "Mat destroy");
	if(mf_flg) {
		ierr = MatDestroy(&isol.A); 
		petsc_throw(ierr, "MatFree mat destroy");
	}

	return tdata;
}

/// Solve a case for a given spatial problem irrespective of whether and what kind of output is needed
int SteadyFlowCase::execute(const Spatial<a_real,NVARS> *const prob, Vec u) const
{
	int ierr = 0;
	
	ierr = execute_starter(prob, u); fvens_throw(ierr, "Startup solve failed!");
	TimingData td = execute_main(prob, u); fvens_throw(ierr, "Steady case solver failed!");
	if(!td.converged)
		throw Tolerance_error("Main flow solve did not converge!");

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
	
	// physical configuration
	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	// simpler numerics for startup
	const FlowNumericsConfig nconfstart = firstorder_spatial_numerics_config(opts);

	const UMesh2dh<a_real> *const m = prob->mesh();

	std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<a_real,NVARS> *const startprob
		= create_const_flowSpatialDiscretization(m, pconf, nconfstart);

	std::cout << "***\n";

	/* NOTE: Since the "startup" solver (meant to generate an initial solution) and the "main" solver
	 * have the same number of unknowns and use first-order Jacobians, we have just one set of
	 * solution vector, Jacobian matrix, preconditioning matrix and KSP solver.
	 */

	// Initialize Jacobian for implicit schemes
	Mat M;
	ierr = setupSystemMatrix<NVARS>(m, &M); CHKERRQ(ierr);
	//ierr = MatCreateVecs(M, u, NULL); CHKERRQ(ierr);

	// setup matrix-free Jacobian if requested
	Mat A;
	MatrixFreeSpatialJacobian<NVARS> mfjac;
	PetscBool mf_flg = PETSC_FALSE;
	ierr = PetscOptionsHasName(NULL, NULL, "-matrix_free_jacobian", &mf_flg); CHKERRQ(ierr);
	if(mf_flg) {
		std::cout << " Allocating matrix-free Jac\n";
		ierr = setup_matrixfree_jacobian<NVARS>(m, &mfjac, &A); 
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

	const SteadySolverConfig temptconf = {false, opts.logfile, opts.firstinitcfl, opts.firstendcfl,
		opts.firstrampstart, opts.firstrampend, opts.firsttolerance, 1};

	SteadySolver<NVARS> * starttime = nullptr, * time = nullptr;

	if(opts.pseudotimetype == "IMPLICIT")
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
	//startprob->initializeUnknowns(*u);

	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

	// setup BLASTed preconditioning if requested
#ifdef USE_BLASTED
	Blasted_data_list bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(ksp,u,startprob,bctx); CHKERRQ(ierr);
	}
#endif

	std::cout << "***\n";

	// computation

	if(opts.usestarter != 0) {

		mfjac.set_spatial(startprob);

		// solve the starter problem to get the initial solution
		ierr = starttime->solve(u); CHKERRQ(ierr);
	}

	// Reset the KSP - could be advantageous for some types of algebraic solvers
	//  This will also reset the BLASTed timing counters.
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
#ifdef USE_BLASTED
	destroyBlastedDataList(&bctx);
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
	bctx = newBlastedDataList();
	if(opts.pseudotimetype == "IMPLICIT") {
		ierr = setup_blasted<NVARS>(ksp,u,startprob,bctx); CHKERRQ(ierr);
	}
#endif

	// setup nonlinear ODE solver for main solve - MUST be done AFTER KSPCreate
	if(opts.pseudotimetype == "IMPLICIT")
	{
		time = new SteadyBackwardEulerSolver<NVARS>(prob, maintconf, ksp);
		std::cout << "\nSet up backward Euler temporal scheme for main solve.\n";
	}
	else
	{
		time = new SteadyForwardEulerSolver<NVARS>(prob, u, maintconf);
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
	destroyBlastedDataList(&bctx);
#endif
	ierr = MatDestroy(&M); CHKERRQ(ierr);
	if(mf_flg) {
		ierr = MatDestroy(&A); 
		CHKERRQ(ierr);
	}

	delete startprob;

	return ierr;
}

}

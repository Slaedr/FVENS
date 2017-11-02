#include <blockmatrices.hpp>
#include "autilities.hpp"
#include "aoutput.hpp"
#include "aodesolver.hpp"
#include "afactory.hpp"

using namespace amat;
using namespace std;
using namespace acfd;

int main(const int argc, const char *const argv[])
{
	// Read control file
	
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);

	// Set up mesh

	UMesh2dh m;
	m.readMesh(opts.meshfile);
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();
	m.compute_periodic_map(opts.periodic_marker, opts.periodic_axis);

	std::cout << "\n***\n";

	// set up problem
	
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
	
	std::cout << "Setting up main spatial scheme.\n";
	const Spatial<NVARS> *const prob = create_const_flowSpatialDiscretization(&m, pconf, nconfmain);
	
	std::cout << "\nSetting up spatial scheme for the initial guess.\n";
	const Spatial<NVARS> *const startprob = create_const_flowSpatialDiscretization(&m, pconf, nconfstart);
	
	std::cout << "\n***\n";
	
	// solution vector
	MVector u(m.gnelem(),NVARS);

	// Initialize Jacobian for implicit schemes; no storage allocated here
	blasted::LinearOperator<a_real,a_int> * M = nullptr;
	if(opts.mattype == 'd') {
		M = new blasted::DLUMatrix<NVARS>(&m,opts.nbuildsweeps,opts.napplysweeps);
		std::cout << " Selected DLU matrix for Jacobian storage, if required.\n";
	}
	else if(opts.mattype == 'c') {
		M = new blasted::BSRMatrix<a_real,a_int,1>(opts.nbuildsweeps,opts.napplysweeps);
		std::cout << " Selected CSR matrix for Jacobian storage, if required.\n";
	}
	else {
		M = new blasted::BSRMatrix<a_real,a_int,NVARS>(opts.nbuildsweeps,opts.napplysweeps);
		std::cout << " Selected BSR matrix for Jacobian storage, if required.\n";
	}

	// set up time discrization
	
	const SteadySolverConfig maintconf {
		opts.lognres, opts.logfile,
		opts.initcfl, opts.endcfl, opts.rampstart, opts.rampend,
		opts.tolerance, opts.maxiter,
		opts.lintol, opts.linmaxiterstart, opts.linmaxiterend, opts.linsolver, opts.restart_vecs,
		opts.preconditioner
	};
	
	const SteadySolverConfig starttconf {
		opts.lognres, opts.logfile,
		opts.firstinitcfl, opts.firstendcfl, opts.firstrampstart, opts.firstrampend,
		opts.firsttolerance, opts.firstmaxiter,
		opts.lintol, opts.linmaxiterstart, opts.linmaxiterend, opts.linsolver, opts.restart_vecs,
		opts.preconditioner
	};

	SteadySolver<NVARS> * starttime=nullptr, * time=nullptr;

	if(opts.timesteptype == "IMPLICIT") 
	{
		if(opts.use_matrix_free)
			std::cout << "!! Matrix-free not implemented yet! Using matrix-storage instead.\n";
			
		// Pre-allocate storage for Jacobian matrix	
		setupMatrixStorage<NVARS>(&m, opts.mattype, M);
		
		if(opts.usestarter != 0)
			starttime = new SteadyBackwardEulerSolver<4>(startprob, starttconf, M);

		time = new SteadyBackwardEulerSolver<4>(prob, maintconf, M);

		std::cout << "Set up backward Euler temporal scheme.\n";
	}
	else 
	{
		if(opts.residualsmoothing)
			setupLaplacianSmoothingMatrix<NVARS>(&m, M);

		if(opts.usestarter != 0)
			starttime = new SteadyForwardEulerSolver<4>(startprob, starttconf,
					opts.residualsmoothing, M);

		time = new SteadyForwardEulerSolver<4>(prob, maintconf,
				opts.residualsmoothing, M);

		std::cout << "Set up explicit forward Euler temporal scheme.\n";
	}
	
	// Ask the spatial discretization context to initialize flow variables
	startprob->initializeUnknowns(u);
	
	std::cout << "\n***\n";

	// computation
	
	if(opts.usestarter != 0)
		// solve the starter problem to get the initial solution
		starttime->solve(u);

	// Solve the main problem
	time->solve(u);
	
	std::cout << "***\n";

	// export output to VTU

	Array2d<a_real> scalars;
	Array2d<a_real> velocities;
	prob->postprocess_point(u, scalars, velocities);

	string scalarnames[] = {"density", "mach-number", "pressure", "temperature"};
	writeScalarsVectorToVtu_PointData(opts.vtu_output_file, 
			m, scalars, scalarnames, velocities, "velocity");

	// export surface data like pressure coeff etc and volume data as plain text files
	
	IdealGasPhysics phy(opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr);
	FlowOutput out(&m, prob, &phy, opts.alpha);
	
	out.exportSurfaceData(u, opts.lwalls, opts.lothers, opts.surfnameprefix);
	
	if(opts.vol_output_reqd == "YES")
		out.exportVolumeData(u, opts.volnameprefix);

	delete starttime;
	delete time;

	delete prob;
	delete startprob;

	cout << "\n--------------- End --------------------- \n\n";
	return 0;
}

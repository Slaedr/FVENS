#include <iostream>
#include <iomanip>
#include <string>
#include <petscvec.h>

#include "spatial/aoutput.hpp"
#include "mesh/ameshutils.hpp"
#include "ode/aodesolver.hpp"
#include "linalg/alinalg.hpp"
#include "utilities/casesolvers.hpp"
#include "utilities/afactory.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"
#include "isentropicvortex.hpp"

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

using namespace fvens;
using namespace fvens_tests;
namespace po = boost::program_options;
using namespace std::literals::string_literals;

int main(int argc, char *argv[])
{
	StatusCode ierr = 0;
	const char help[] = "Finite volume solver for Euler or Navier-Stokes equations.\n\
		Arguments needed: FVENS control file and PETSc options file with -options_file.\n";

	ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);
	int mpirank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank);

	po::options_description desc
		("FVENS unsteady convergence test.\n"s
		 + " The first argument is the input control file name.\n"
		 + "Further options");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	// Get number of meshes
	//const int nmesh = parsePetscCmd_int("-number_of_meshes");
	const int nmesh = cmdvars["number_of_meshes"].as<int>();
	desc.add_options()("number_of_meshes", "Number of grids for order-of-accuracy test");

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file

	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

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
	infile.close();

	const IsenVortexConfig ivconf {opts.gamma, opts.Minf, vcentre, strength,
			clength, sigma, opts.alpha};
	const IsentropicVortexProblem isen(ivconf);

	std::vector<double> lh(nmesh), lerrors(nmesh), slopes(nmesh-1);

	for(int imesh = 0; imesh < nmesh; imesh++) {
		
		std::string meshi = opts.meshfile + std::to_string(imesh) + ".msh";
		// Set up mesh

		UMesh2dh<a_real> m;
		m.readMesh(meshi);
		int ierr = preprocessMesh(m); 
		fvens_throw(ierr, "Mesh could not be preprocessed!");
		//m.compute_periodic_map(opts.periodic_marker, opts.periodic_axis);
		for(size_t i = 0; i < opts.bcconf.size(); i++)
			if(opts.bcconf[i].bc_type == PERIODIC_BC)
				m.compute_periodic_map(opts.bcconf[i].bc_opts[0], opts.bcconf[i].bc_opts[1]);

		// Check periodic map
		/*m.compute_boundary_maps();
		for(a_int i = 0; i < m.gnbface(); i++) {
			std::cout << m.gbifmap(i) << " -> " << m.gperiodicmap(i) << '\n';
			}*/

		std::cout << "\n***\n";

		std::cout << "Setting up main spatial scheme.\n";
		const Spatial<a_real,NVARS> *const prob
			= create_const_flowSpatialDiscretization(&m, pconf, nconfmain);

		Vec u, uexact;
		ierr = VecCreateSeq(PETSC_COMM_SELF, m.gnelem()*4, &u); CHKERRQ(ierr);
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
		
		// get the FlowFV to compute the entropy error
		const FlowFV_base<a_real>* fprob = reinterpret_cast<const FlowFV_base<a_real>*>(prob);
		//err = fprob->compute_entropy_cell(u);
		IdealGasPhysics<a_real> phy{opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr};
		FlowOutput flowoutput(fprob, &phy, opts.alpha);

		const a_real err = flowoutput.compute_entropy_cell(u);
		const double h = 1.0/sqrt(m.gnelem());
		std::cout << "Log of Mesh size and error are " << log10(h) << "  " << log10(err) << std::endl;
		lh[imesh] = log10(h);
		lerrors[imesh] = log10(err);
		if(imesh > 0)
			slopes[imesh-1] = (lerrors[imesh]-lerrors[imesh-1])/(lh[imesh]-lh[imesh-1]);

		amat::Array2d<a_real> scalars;
		amat::Array2d<a_real> velocities;
		flowoutput.postprocess_point(u, scalars, velocities);

		std::string scalarnames[] = {"density", "mach-number", "pressure", "temperature"};
		writeScalarsVectorToVtu_PointData(opts.vtu_output_file+std::to_string(imesh)+".vtu",
		                                  m, scalars, scalarnames, velocities, "velocity");

		delete prob;
		ierr = VecDestroy(&u); CHKERRQ(ierr);
		ierr = VecDestroy(&uexact); CHKERRQ(ierr);
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

#include "autilities.hpp"
#include "alinalg.hpp"
#include <iostream>
#include <petscmat.h>

namespace acfd {

void open_file_toRead(const std::string file, std::ifstream& fin)
{
	fin.open(file);
	if(!fin) {
		std::cout << "! Could not open file "<< file <<" !\n";
		std::abort();
	}
}

void open_file_toWrite(const std::string file, std::ofstream& fout)
{
	fout.open(file);
	if(!fout) {
		std::cout << "! Could not open file "<< file <<" !\n";
		//std::abort();
	}
}

const FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[])
{
	if(argc < 2)
	{
		std::cerr << "! Please give a control file name.\n";
		std::abort();
	}

	FlowParserOptions opts;

	// set some default values
	opts.useconstvisc = false;
	opts.viscsim = false;
	opts.order2 = true;
	opts.Reinf=0; opts.Tinf=0; opts.Pr=0; 
	opts.twalltemp=0; opts.twallvel = 0;
	opts.tpwalltemp=0; opts.tpwallpressure=0; opts.tpwallvel=0; 
	opts.adiawallvel = 0;
	opts.isothermalwall_marker=-1; 
	opts.isothermalpressurewall_marker=-1; 
	opts.adiabaticwall_marker=-1; 

	std::ifstream control; 
	open_file_toRead(argv[1], control);

	std::string dum; char dumc;

	std::getline(control,dum); control >> opts.meshfile;
	if(opts.meshfile == "READFROMCMD")
	{
		if(argc >= 3)
			opts.meshfile = argv[2];
		else {
			std::cout << "! Mesh file not given in command line!\n";
			std::abort();
		}
	}

	control >> dum; control >> opts.vtu_output_file;
	control >> dum; control >> opts.logfile;

	std::string lognresstr;
	control >> dum; control >> lognresstr;
	if(lognresstr == "YES")
		opts.lognres = true;
	else
		opts.lognres = false;

	control >> dum;
	control >> dum; control >> opts.simtype;
	control >> dum; control >> opts.gamma;
	control >> dum; control >> opts.alpha;
	opts.alpha = opts.alpha*PI/180.0;
	control >> dum; control >> opts.Minf;
	if(opts.simtype == "NAVIERSTOKES") {
		opts.viscsim = true;

		control >> dum; control >> opts.Tinf;
		control >> dum; control >> opts.Reinf;
		control >> dum; control >> opts.Pr;

		std::string constvisc;
		control >> dum; control >> constvisc;
		if(constvisc == "YES")
			opts.useconstvisc = true;
	}
	control >> dum; control >> opts.soln_init_type;
	if(opts.soln_init_type == 1) {
		control >> dum; control >> opts.init_soln_file;
	}
	control.get(dumc); std::getline(control,dum); // FIXME formatting of control file
	control >> dum; control >> opts.slipwall_marker;
	control >> dum; control >> opts.farfield_marker;
	control >> dum; control >> opts.inout_marker;
	control >> dum; control >> opts.extrap_marker;
	control >> dum; control >> opts.periodic_marker;
	if(opts.periodic_marker >= 0) {
		control >> dum; control >> opts.periodic_axis;
	}
	if(opts.viscsim) {
		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.isothermalwall_marker;

		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.twalltemp >> opts.twallvel;

		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.adiabaticwall_marker;

		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.adiawallvel;

		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.isothermalpressurewall_marker;

		std::getline(control,dum); std::getline(control,dum); 
		control >> opts.tpwalltemp >> opts.tpwallvel >> opts.tpwallpressure;
	}

	control >> dum; control >> opts.num_out_walls;
	opts.lwalls.resize(opts.num_out_walls);
	if(opts.num_out_walls > 0) {
		control >> dum;
		for(int i = 0; i < opts.num_out_walls; i++)
			control >> opts.lwalls[i];
	}

	control >> dum; control >> opts.num_out_others;
	opts.lothers.resize(opts.num_out_others);
	if(opts.num_out_others > 0)
	{
		control >> dum;
		for(int i = 0; i < opts.num_out_others; i++)
			control >> opts.lothers[i];
	}

	if(opts.num_out_others > 0 || opts.num_out_walls > 0) {
		control >> dum; 
		control >> opts.surfnameprefix;
	}
	control >> dum; control >> opts.vol_output_reqd;
	if(opts.vol_output_reqd == "YES") {
		control >> dum; 
		control >> opts.volnameprefix;
	}

	control >> dum;
	control >> dum; control >> opts.invflux;
	control >> dum; control >> opts.gradientmethod;
	if(opts.gradientmethod == "NONE")
		opts.order2 = false;

	control >> dum; control >> opts.limiter;

	std::string recprim;
	control >> dum; control >> recprim;

	control >> dum;
	control >> dum; control >> opts.timesteptype;
	control >> dum; control >> opts.initcfl;
	control >> dum; control >> opts.endcfl;
	control >> dum; control >> opts.rampstart >> opts.rampend;
	control >> dum; control >> opts.tolerance;
	control >> dum; control >> opts.maxiter;
	control >> dum;
	control >> dum; control >> opts.usestarter;
	control >> dum; control >> opts.firstinitcfl;
	control >> dum; control >> opts.firstendcfl;
	control >> dum; control >> opts.firstrampstart >> opts.firstrampend;
	control >> dum; control >> opts.firsttolerance;
	control >> dum; control >> opts.firstmaxiter;
	if(opts.timesteptype == "IMPLICIT") {
		control >> dum;
		control >> dum; control >> opts.invfluxjac;
	}
	control.close();
	
	// check for some PETSc options
	char petsclogfile[200];
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetString(NULL, NULL, "-fvens_log_file", petsclogfile, 200, &set);
	if(set)
		opts.logfile = petsclogfile;

	return opts;
}

FlowPhysicsConfig extract_spatial_physics_config(const FlowParserOptions& opts)
{
	const FlowPhysicsConfig pconf { 
		opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr, opts.alpha,
		opts.viscsim, opts.useconstvisc,
		opts.isothermalwall_marker, opts.adiabaticwall_marker, opts.isothermalpressurewall_marker,
		opts.slipwall_marker, opts.farfield_marker, opts.inout_marker, 
		opts.extrap_marker, opts.periodic_marker,
		opts.twalltemp, opts.twallvel, opts.adiawallvel, opts.tpwalltemp, opts.tpwallvel
	};
	return pconf;
}

FlowNumericsConfig extract_spatial_numerics_config(const FlowParserOptions& opts)
{
	const FlowNumericsConfig nconf {opts.invflux, opts.invfluxjac, 
		opts.gradientmethod, opts.limiter, opts.order2};
	return nconf;
}

StatusCode reorderMesh(const char *const ordering, const Spatial<1>& sd, UMesh2dh& m)
{
	// The implementation must be changed for the multi-process case
	
	Mat A;
	CHKERRQ(MatCreate(PETSC_COMM_SELF, &A));
	CHKERRQ(MatSetType(A, MATSEQAIJ));
	CHKERRQ(setJacobianPreallocation<1>(&m, A));

	Vec u;
	CHKERRQ(VecCreate(PETSC_COMM_SELF, &u));
	CHKERRQ(VecSetSizes(u, m.gnelem(), m.gnelem()));
	CHKERRQ(VecSet(u,1.0));

	CHKERRQ(sd.compute_jacobian(u, A));

	IS rperm, cperm;
	const PetscInt *rinds, *cinds;
	CHKERRQ(MatGetOrdering(A, ordering, &rperm, &cperm));
	CHKERRQ(ISGetIndices(rperm, &rinds));
	CHKERRQ(ISGetIndices(cperm, &cinds));
	// check for symmetric permutation
	for(a_int i = 0; i < m.gnelem(); i++)
		assert(rinds[i] == cinds[i]);

	m.reorder_cells(rinds);
	
	CHKERRQ(ISRestoreIndices(rperm, &rinds));
	CHKERRQ(MatDestroy(&A));
	CHKERRQ(VecDestroy(&u));
	return 0;
}

StatusCode preprocessMesh(UMesh2dh& m)
{
	char ordstr[PETSCOPTION_STR_LEN];
	PetscBool flag = PETSC_FALSE;
	CHKERRQ(PetscOptionsGetString(NULL, NULL, "-mesh_reorder", ordstr, PETSCOPTION_STR_LEN, &flag));
	if(flag == PETSC_FALSE) {
		std::cout << "No reordering requested.\n";
	}
	else {
		m.compute_topological();

		DiffusionMA<1> sd(&m, 1.0, 0.0, 
			[](const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm)
			{ sourceterm[0] = 0; }, 
		"NONE");

		CHKERRQ(reorderMesh(ordstr, sd, m));
	}
		
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();

	return 0;
}

}

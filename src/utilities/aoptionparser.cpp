/** \file aoptionparser.cpp
 * \brief Parse options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#include "aoptionparser.hpp"
#include "aerrorhandling.hpp"
#include <iostream>

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
		throw std::runtime_error("Could not open file " + file);
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
	control >> dum; control >> opts.limiter_param;

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
		opts.gradientmethod, opts.limiter, opts.limiter_param, opts.order2};
	return nconf;
}

FlowNumericsConfig firstorder_spatial_numerics_config(const FlowParserOptions& opts)
{
	const FlowNumericsConfig nconf {opts.invflux, opts.invfluxjac, 
		"NONE", "NONE", 1.0 , false};
	return nconf;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
int parsePetscCmd_int(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	int output = 0;
	ierr = PetscOptionsGetInt(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get int ")+ optionname);
	fvens_throw(!set, std::string("Int ") + optionname + std::string(" not set"));
	return output;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
PetscReal parseOptionalPetscCmd_real(const std::string optionname, const PetscReal defval)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscReal output = 0;
	ierr = PetscOptionsGetReal(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get real ")+ optionname);
	if(!set) {
		std::cout << "PETSc cmd option " << optionname << " not set; using default.\n";
		output = defval;
	}
	return output;
}

bool parsePetscCmd_bool(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscBool output = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL, NULL, optionname.c_str(), &output, &set);
	petsc_throw(ierr, std::string("Could not get bool ")+ optionname);
	fvens_throw(!set, std::string("Bool ") + optionname + std::string(" not set"));
	return (bool)output;
}


std::string parsePetscCmd_string(const std::string optionname, const size_t p_strlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	char* tt = new char[p_strlen+1];
	ierr = PetscOptionsGetString(NULL, NULL, optionname.data(), tt, p_strlen, &set);
	petsc_throw(ierr, std::string("Could not get string ") + std::string(optionname));
	fvens_throw(!set, std::string("String ") + optionname + std::string(" not set"));
	const std::string stropt = tt;
	delete [] tt;
	return stropt;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	arr.resize(len);

	petsc_throw(ierr, std::string("Could not get array ") + std::string(optionname));
	fvens_throw(!set, std::string("Array ") + optionname + std::string(" not set"));
	return arr;
}

std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	arr.resize(len);

	petsc_throw(ierr, std::string("Could not get array ") + std::string(optionname));
	if(!set) {
		std::cout << "Array " << optionname << " not set.\n";
		arr.resize(0);
	}
	return arr;
}

}

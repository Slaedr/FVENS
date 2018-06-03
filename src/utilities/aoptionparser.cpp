/** \file aoptionparser.cpp
 * \brief Parse options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#include "aoptionparser.hpp"
#include "aerrorhandling.hpp"
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

namespace acfd {

namespace po = boost::program_options;
namespace pt = boost::property_tree;

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

po::variables_map parse_cmd_options(const int argc, const char *const argv[],
                                    po::options_description& desc)
{
	desc.add_options()
		("help", "Help message")
		("mesh_file", po::value<std::string>(),
		 "Mesh file to solve the problem on; overrides the corresponding option in the control file");

	po::variables_map cmdvarmap;
	po::parsed_options parsedopts =
		po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
	po::store(parsedopts, cmdvarmap);
	po::notify(cmdvarmap);

	return cmdvarmap;
}

/// Extract a string from a property tree, convert it to upper case and return it
static inline std::string get_upperCaseString(const pt::ptree& tree, const std::string path) {
	return boost::to_upper_copy<std::string>(tree.get<std::string>(path));
}

/// Extract a vector of things from a string
/** Assumes a space-separated list in the input string.
 */
template <typename T> static inline
std::vector<T> parseStringToVector(const std::string str)
{
	T elem;
	std::vector<T> vec;
	std::stringstream ss(str);
	while(ss >> elem) {
		vec.push_back(elem);
	}
	return vec;
}

FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[],
                                         const po::variables_map cmdvars)
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
	opts.time_integrator = "NONE";

	// Define string constants used as keywords in the control file

	// categories
	const std::string c_io = "io";
	const std::string c_flowconds = "flow-conditions";
	const std::string c_bcs = "bc";
	const std::string c_phy_time = "time";
	const std::string c_spatial = "spatial-discretization";
	const std::string c_pseudotime = "pseudotime";

	// options that possibly repeat
	// BCs
	const std::string slipwall_m = "slipwall_marker";
	const std::string farfield_m = "farfield_marker";
	const std::string inoutflow_m = "inflow_outflow_marker";
	const std::string extrap_m = "extrapolation_marker";
	const std::string periodic_m = "periodic_marker";
	const std::string periodic_axis = "periodic_axis";
	// Pseudotime settings
	const std::string pt_step_type = "pseudotime_stepping_type";
	const std::string pt_cflmin = "cfl_min";
	const std::string pt_cflmax = "cfl_max";
	const std::string pt_tol = "tolerance";
	const std::string pt_main = "main";
	const std::string pt_init = "initialization";

	// start reading control file and override with command line options when applicable

	pt::ptree infopts;
	pt::read_info(argv[1], infopts);

	// std::ifstream control; 
	// open_file_toRead(argv[1], control);

	// std::string dum; char dumc;

	//std::getline(control,dum); control >> opts.meshfile;
	opts.meshfile = infopts.get<std::string>("io.mesh_file");
	if(cmdvars.count("mesh_file")) {
		std::cout << "Read mesh file from the command line rather than the control file.\n";
		opts.meshfile = cmdvars["mesh_file"].as<std::string>();
	}

	// control >> dum; control >> opts.vtu_output_file;
	// control >> dum; control >> opts.logfile;
	opts.vtu_output_file = infopts.get<std::string>(c_io+".solution_output_file");
	opts.logfile = infopts.get<std::string>(c_io+".log_file_prefix");

	// std::string lognresstr;
	// control >> dum; control >> lognresstr;
	// if(lognresstr == "YES")
	// 	opts.lognres = true;
	// else
	// 	opts.lognres = false;
	opts.lognres = infopts.get<bool>(c_io+".convergence_history_required");

	// control >> dum;
	// control >> dum; control >> opts.flowtype;
	// control >> dum; control >> opts.gamma;
	// control >> dum; control >> opts.alpha;
	// opts.alpha = opts.alpha*PI/180.0;
	// control >> dum; control >> opts.Minf;
	opts.flowtype = get_upperCaseString(infopts, c_flowconds+".flow_type");
	opts.gamma = infopts.get<a_real>(c_flowconds+".adiabatic_index");
	opts.alpha = PI/180.0*infopts.get<a_real>(c_flowconds+".angle_of_attack");
	opts.Minf = infopts.get<a_real>(c_flowconds+".freestream_Mach_number");
	if(opts.flowtype == "NAVIERSTOKES") {
		opts.viscsim = true;

		// control >> dum; control >> opts.Tinf;
		// control >> dum; control >> opts.Reinf;
		// control >> dum; control >> opts.Pr;
		opts.Tinf = infopts.get<a_real>(c_flowconds+".freestream_temperature");
		opts.Reinf = infopts.get<a_real>(c_flowconds+".freestream_Reynolds_number");
		opts.Pr = infopts.get<a_real>(c_flowconds+".Prandtl_number");

		// std::string constvisc;
		// control >> dum; control >> constvisc;
		// if(constvisc == "YES")
		// 	opts.useconstvisc = true;
		opts.useconstvisc = infopts.get(c_flowconds+".use_constant_viscosity",false);
	}

	// control >> dum; control >> opts.soln_init_type;
	// if(opts.soln_init_type == 1) {
	// 	control >> dum; control >> opts.init_soln_file;
	// }
	// control.get(dumc); std::getline(control,dum); // FIXME formatting of control file
	// control >> dum; control >> opts.slipwall_marker;
	// control >> dum; control >> opts.farfield_marker;
	// control >> dum; control >> opts.inout_marker;
	// control >> dum; control >> opts.extrap_marker;
	// control >> dum; control >> opts.periodic_marker;
	opts.slipwall_marker = infopts.get(c_bcs+"."+slipwall_m,-1);
	opts.farfield_marker = infopts.get(c_bcs+"."+farfield_m,-1);
	opts.inout_marker = infopts.get(c_bcs+"."+inoutflow_m,-1);
	opts.extrap_marker = infopts.get(c_bcs+"."+extrap_m,-1);
	opts.periodic_marker = infopts.get(c_bcs+"."+periodic_m,-1);
	if(opts.periodic_marker >= 0) {
		// control >> dum; control >> opts.periodic_axis;
		opts.periodic_axis = infopts.get<int>(c_bcs+"."+periodic_axis);
	}
	if(opts.viscsim) {
		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.isothermalwall_marker;

		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.twalltemp >> opts.twallvel;

		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.adiabaticwall_marker;

		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.adiawallvel;

		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.isothermalpressurewall_marker;

		// std::getline(control,dum); std::getline(control,dum); 
		// control >> opts.tpwalltemp >> opts.tpwallvel >> opts.tpwallpressure;
	}

	// control >> dum; control >> opts.num_out_walls;
	// opts.num_out_walls = infopts.get<int>(c_bcs+".numberof_output_wall_boundaries");
	// opts.lwalls.resize(opts.num_out_walls);
	// if(opts.num_out_walls > 0) {
	// 	control >> dum;
	// 	for(int i = 0; i < opts.num_out_walls; i++)
	// 		control >> opts.lwalls[i];
	// }
	auto optlwalls = infopts.get_optional<std::string>(c_bcs+".listof_output_wall_boundaries");
	if(optlwalls)
		opts.lwalls = parseStringToVector<int>(*optlwalls);
	opts.num_out_walls = static_cast<int>(opts.lwalls.size());

	// control >> dum; control >> opts.num_out_others;
	// opts.lothers.resize(opts.num_out_others);
	// if(opts.num_out_others > 0)
	// {
	// 	control >> dum;
	// 	for(int i = 0; i < opts.num_out_others; i++)
	// 		control >> opts.lothers[i];
	// }
	auto optlothers= infopts.get_optional<std::string>(c_bcs+".listof_output_other_boundaries");
	if(optlothers)
		opts.lothers = parseStringToVector<int>(*optlothers);
	opts.num_out_others = static_cast<int>(opts.lothers.size());

	if(opts.num_out_others > 0 || opts.num_out_walls > 0) {
		// control >> dum; 
		// control >> opts.surfnameprefix;
		opts.surfnameprefix = infopts.get<std::string>(c_bcs+".surface_output_file_prefx");
	}
	// control >> dum; control >> opts.vol_output_reqd;
	// if(opts.vol_output_reqd == "YES") {
	// 	control >> dum; 
	// 	control >> opts.volnameprefix;
	// }
	auto volnamepref = infopts.get_optional<std::string>(c_bcs+".volume_output_file_prefix");
	if(volnamepref) {
		opts.volnameprefix = *volnamepref;
		opts.vol_output_reqd = "YES";
	} else {
		opts.vol_output_reqd = "NO";
	}

	// Time stuff
	// control.get(dumc); std::getline(control,dum);
	// std::getline(control,dum); control >> opts.sim_type; control.get(dumc);
	opts.sim_type = get_upperCaseString(infopts, c_phy_time+".simulation-type");
	if(opts.sim_type == "UNSTEADY")
	{
		// std::getline(control,dum); control >> opts.final_time; control.get(dumc);
		// std::getline(control,dum); control >> opts.time_integrator; control.get(dumc);
		// std::getline(control,dum); control >> opts.time_order; control.get(dumc);
		// std::getline(control,dum); control >> opts.phy_cfl; control.get(dumc);
		opts.final_time = infopts.get<a_real>(c_phy_time+".final_time");
		opts.time_integrator = get_upperCaseString(infopts, c_phy_time + ".time_integrator");
		opts.time_order = infopts.get<int>(c_phy_time + ".temporal_order");

		if(opts.time_integrator == "TVDRK")
			opts.phy_cfl = infopts.get<a_real>(c_phy_time+".physical_cfl");
		else
			opts.phy_timestep = infopts.get<a_real>(c_phy_time+".physical_time_step");
	}

	// control >> dum;
	// control >> dum; control >> opts.invflux;
	// control >> dum; control >> opts.gradientmethod;

	// control >> dum; control >> opts.limiter;
	// control >> dum; control >> opts.limiter_param;
	opts.invflux = get_upperCaseString(infopts, c_spatial+".inviscid_flux");
	opts.gradientmethod = get_upperCaseString(infopts, c_spatial+".gradient_method");
	if(opts.gradientmethod == "NONE")
		opts.order2 = false;
	opts.limiter = get_upperCaseString(infopts, c_spatial+".limiter");

	// control >> dum;
	// control >> dum; control >> opts.pseudotimetype;
	// control >> dum; control >> opts.initcfl;
	// control >> dum; control >> opts.endcfl;
	// control >> dum; control >> opts.rampstart >> opts.rampend;
	// control >> dum; control >> opts.tolerance;
	// control >> dum; control >> opts.maxiter;
	// control >> dum;
	// control >> dum; control >> opts.usestarter;
	// control >> dum; control >> opts.firstinitcfl;
	// control >> dum; control >> opts.firstendcfl;
	// control >> dum; control >> opts.firstrampstart >> opts.firstrampend;
	// control >> dum; control >> opts.firsttolerance;
	// control >> dum; control >> opts.firstmaxiter;
	opts.pseudotimetype = get_upperCaseString(infopts, c_pseudotime+".pseudotime_stepping_type");

	opts.initcfl = infopts.get<a_real>(c_pseudotime+"."+pt_main+".cfl_min");
	opts.endcfl = infopts.get<a_real>(c_pseudotime+"."+pt_main+".cfl_max");
	opts.tolerance = infopts.get<a_real>(c_pseudotime+"."+pt_main+".tolerance");
	opts.maxiter = infopts.get<int>(c_pseudotime+"."+pt_main+".max_timesteps");
	if(infopts.find(c_pseudotime+"."+pt_init) != infopts.not_found()) {
		opts.usestarter = 1;
		opts.firstinitcfl = infopts.get<a_real>(c_pseudotime+"."+pt_init+".cfl_min");
		opts.firstendcfl = infopts.get<a_real>(c_pseudotime+"."+pt_init+".cfl_max");
		opts.firsttolerance = infopts.get<a_real>(c_pseudotime+"."+pt_init+".tolerance");
		opts.firstmaxiter = infopts.get<int>(c_pseudotime+"."+pt_init+".max_timesteps");
	}

	if(opts.pseudotimetype == "IMPLICIT") {
		// control >> dum;
		// control >> dum; control >> opts.invfluxjac;
		opts.invfluxjac = get_upperCaseString(infopts, "Jacobian_inviscid_flux");
	}
	
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

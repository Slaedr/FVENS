/** \file controlparser.cpp
 * \brief Control file parsing
 * \author Aditya Kashi
 * \date 2017-10
 */

#include "controlparser.hpp"
#include "aerrorhandling.hpp"
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

namespace fvens {

namespace po = boost::program_options;
namespace pt = boost::property_tree;

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
	const std::string c_flowconds = "flow_conditions";
	const std::string c_bcs = "bc";
	const std::string c_phy_time = "time";
	const std::string c_spatial = "spatial_discretization";
	const std::string c_pseudotime = "pseudotime";

	// options that possibly repeat
	// BCs
	const std::string slipwall_m = "slipwall_marker";
	const std::string farfield_m = "farfield_marker";
	const std::string inoutflow_m = "inflow_outflow_marker";
	const std::string extrap_m = "extrapolation_marker";
	const std::string periodic_m = "periodic_marker";
	const std::string periodic_axis = "periodic_axis";
	const std::string isothermalwall_m = "isothermal_wall_marker";
	const std::string adiabaticwall_m = "adiabatic_wall_marker";
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

	opts.meshfile = infopts.get<std::string>("io.mesh_file");
	if(cmdvars.count("mesh_file")) {
		std::cout << "Read mesh file from the command line rather than the control file.\n";
		opts.meshfile = cmdvars["mesh_file"].as<std::string>();
	}

	opts.vtu_output_file = infopts.get<std::string>(c_io+".solution_output_file");
	opts.logfile = infopts.get<std::string>(c_io+".log_file_prefix");
	opts.lognres = infopts.get<bool>(c_io+".convergence_history_required");

	opts.flowtype = get_upperCaseString(infopts, c_flowconds+".flow_type");
	opts.gamma = infopts.get<a_real>(c_flowconds+".adiabatic_index");
	opts.alpha = PI/180.0*infopts.get<a_real>(c_flowconds+".angle_of_attack");
	opts.Minf = infopts.get<a_real>(c_flowconds+".freestream_Mach_number");
	if(opts.flowtype == "NAVIERSTOKES") {
		opts.viscsim = true;
		opts.Tinf = infopts.get<a_real>(c_flowconds+".freestream_temperature");
		opts.Reinf = infopts.get<a_real>(c_flowconds+".freestream_Reynolds_number");
		opts.Pr = infopts.get<a_real>(c_flowconds+".Prandtl_number");
		opts.useconstvisc = infopts.get(c_flowconds+".use_constant_viscosity",false);
	}

	opts.slipwall_marker = infopts.get(c_bcs+"."+slipwall_m,-1);
	opts.farfield_marker = infopts.get(c_bcs+"."+farfield_m,-1);
	opts.inout_marker = infopts.get(c_bcs+"."+inoutflow_m,-1);
	opts.extrap_marker = infopts.get(c_bcs+"."+extrap_m,-1);
	opts.periodic_marker = infopts.get(c_bcs+"."+periodic_m,-1);
	if(opts.periodic_marker >= 0) {
		opts.periodic_axis = infopts.get<int>(c_bcs+"."+periodic_axis);
	}
	if(opts.viscsim) {
		opts.isothermalwall_marker = infopts.get(c_bcs+"."+isothermalwall_m,-1);
		if(opts.isothermalwall_marker >= 0) {
			opts.twalltemp = infopts.get<a_real>(c_bcs+".isothermal_wall_temperature");
			opts.twallvel = infopts.get<a_real>(c_bcs+".isothermal_wall_velocity");
		}
		opts.adiabaticwall_marker = infopts.get(c_bcs+"."+adiabaticwall_m,-1);
		if(opts.adiabaticwall_marker >= 0)
			opts.adiawallvel = infopts.get<a_real>(c_bcs+".adiabatic_wall_velocity");
	}

	auto optlwalls = infopts.get_optional<std::string>(c_bcs+".listof_output_wall_boundaries");
	if(optlwalls)
		opts.lwalls = parseStringToVector<int>(*optlwalls);
	opts.num_out_walls = static_cast<int>(opts.lwalls.size());

	auto optlothers= infopts.get_optional<std::string>(c_bcs+".listof_output_other_boundaries");
	if(optlothers)
		opts.lothers = parseStringToVector<int>(*optlothers);
	opts.num_out_others = static_cast<int>(opts.lothers.size());

	if(opts.num_out_others > 0 || opts.num_out_walls > 0) {
		opts.surfnameprefix = infopts.get<std::string>(c_bcs+".surface_output_file_prefix");
	}

	auto volnamepref = infopts.get_optional<std::string>(c_bcs+".volume_output_file_prefix");
	if(volnamepref) {
		opts.volnameprefix = *volnamepref;
		opts.vol_output_reqd = "YES";
	} else {
		opts.vol_output_reqd = "NO";
	}

	// Time stuff
	opts.sim_type = get_upperCaseString(infopts, c_phy_time+".simulation_type");
	if(opts.sim_type == "UNSTEADY")
	{
		opts.final_time = infopts.get<a_real>(c_phy_time+".final_time");
		opts.time_integrator = get_upperCaseString(infopts, c_phy_time + ".time_integrator");
		opts.time_order = infopts.get<int>(c_phy_time + ".temporal_order");

		if(opts.time_integrator == "TVDRK")
			opts.phy_cfl = infopts.get<a_real>(c_phy_time+".physical_cfl");
		else
			opts.phy_timestep = infopts.get<a_real>(c_phy_time+".physical_time_step");
	}

	opts.invflux = get_upperCaseString(infopts, c_spatial+".inviscid_flux");
	opts.gradientmethod = get_upperCaseString(infopts, c_spatial+".gradient_method");
	if(opts.gradientmethod == "NONE")
		opts.order2 = false;
	opts.limiter = get_upperCaseString(infopts, c_spatial+".limiter");

	opts.pseudotimetype = get_upperCaseString(infopts, c_pseudotime+".pseudotime_stepping_type");

	opts.initcfl = infopts.get<a_real>(c_pseudotime+"."+pt_main+".cfl_min");
	opts.endcfl = infopts.get<a_real>(c_pseudotime+"."+pt_main+".cfl_max");
	opts.tolerance = infopts.get<a_real>(c_pseudotime+"."+pt_main+".tolerance");
	opts.maxiter = infopts.get<int>(c_pseudotime+"."+pt_main+".max_timesteps");
	if(infopts.get_child_optional(c_pseudotime+"."+pt_init)) {
		opts.usestarter = 1;
		opts.firstinitcfl = infopts.get<a_real>(c_pseudotime+"."+pt_init+".cfl_min");
		opts.firstendcfl = infopts.get<a_real>(c_pseudotime+"."+pt_init+".cfl_max");
		opts.firsttolerance = infopts.get<a_real>(c_pseudotime+"."+pt_init+".tolerance");
		opts.firstmaxiter = infopts.get<int>(c_pseudotime+"."+pt_init+".max_timesteps");
	}

	if(opts.pseudotimetype == "IMPLICIT") {
		opts.invfluxjac = get_upperCaseString(infopts, "Jacobian_inviscid_flux");
		if(opts.invfluxjac == "CONSISTENT")
			opts.invfluxjac = opts.invflux;
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

}

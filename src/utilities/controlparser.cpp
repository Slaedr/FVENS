/** \file controlparser.cpp
 * \brief Control file parsing
 * \author Aditya Kashi
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "controlparser.hpp"
#include "aerrorhandling.hpp"
#include "spatial/abctypemap.hpp"
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

/// Parse options related to boundary conditions
static std::vector<FlowBCConfig> parse_BC_options(const pt::ptree& infopts,
                                                  const std::string bc_keyword);

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

	if(cmdvars.count("write_final_linear_system")) {
		opts.write_final_lin_sys = cmdvars["write_final_linear_system"].as<bool>();
		if(opts.write_final_lin_sys)
			std::cout << " Will write out the final linear system.\n";
	} else {
		opts.write_final_lin_sys = false;
	}

	opts.vtu_output_file = infopts.get<std::string>(c_io+".solution_output_file");
	opts.logfile = infopts.get<std::string>(c_io+".log_file_prefix");
	opts.lognres = infopts.get<bool>(c_io+".convergence_history_required");

	opts.flowtype = get_upperCaseString(infopts, c_flowconds+".flow_type");
	opts.gamma = infopts.get<freal>(c_flowconds+".adiabatic_index");
	opts.alpha = PI/180.0*infopts.get<freal>(c_flowconds+".angle_of_attack");
	opts.Minf = infopts.get<freal>(c_flowconds+".freestream_Mach_number");
	if(opts.flowtype == "NAVIERSTOKES" || opts.flowtype == "RANS") {
		opts.viscsim = true;
		opts.Tinf = infopts.get<freal>(c_flowconds+".freestream_temperature");
		opts.Reinf = infopts.get<freal>(c_flowconds+".freestream_Reynolds_number");
		opts.Pr = infopts.get<freal>(c_flowconds+".Prandtl_number");
		opts.useconstvisc = infopts.get(c_flowconds+".use_constant_viscosity",false);
		if(opts.flowtype == "RANS")
			throw UnsupportedOptionError("flowtype RANS");
	}
	else {
		opts.viscsim = false;
		opts.Tinf = 298.0;
		opts.Reinf = 1.0/0.0;
		opts.Pr = 0.0/0.0;
	}

	opts.bcconf = parse_BC_options(infopts, c_bcs);

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
		opts.final_time = infopts.get<freal>(c_phy_time+".final_time");
		opts.time_integrator = get_upperCaseString(infopts, c_phy_time + ".time_integrator");
		opts.time_order = infopts.get<int>(c_phy_time + ".temporal_order");

		if(opts.time_integrator == "TVDRK")
			opts.phy_cfl = infopts.get<freal>(c_phy_time+".physical_cfl");
		else
			opts.phy_timestep = infopts.get<freal>(c_phy_time+".physical_time_step");
	}

	opts.invflux = get_upperCaseString(infopts, c_spatial+".inviscid_flux");
	opts.gradientmethod = get_upperCaseString(infopts, c_spatial+".gradient_method");
	if(opts.gradientmethod == "NONE")
		opts.order2 = false;
	opts.limiter = get_upperCaseString(infopts, c_spatial+".limiter");

	opts.pseudotimetype = get_upperCaseString(infopts, c_pseudotime+".pseudotime_stepping_type");

	opts.initcfl = infopts.get<freal>(c_pseudotime+"."+pt_main+".cfl_min");
	opts.endcfl = infopts.get<freal>(c_pseudotime+"."+pt_main+".cfl_max");
	opts.tolerance = infopts.get<freal>(c_pseudotime+"."+pt_main+".tolerance");
	opts.maxiter = infopts.get<int>(c_pseudotime+"."+pt_main+".max_timesteps");
	if(infopts.get_child_optional(c_pseudotime+"."+pt_init)) {
		opts.usestarter = 1;
		opts.firstinitcfl = infopts.get<freal>(c_pseudotime+"."+pt_init+".cfl_min");
		opts.firstendcfl = infopts.get<freal>(c_pseudotime+"."+pt_init+".cfl_max");
		opts.firsttolerance = infopts.get<freal>(c_pseudotime+"."+pt_init+".tolerance");
		opts.firstmaxiter = infopts.get<int>(c_pseudotime+"."+pt_init+".max_timesteps");
	}

	if(opts.pseudotimetype == "IMPLICIT") {
		opts.invfluxjac = get_upperCaseString(infopts, "Jacobian_inviscid_flux");
		if(opts.invfluxjac == "CONSISTENT")
			opts.invfluxjac = opts.invflux;

		opts.nl_update_scheme = get_upperCaseString(infopts, c_pseudotime+"."+"nonlinear_update_scheme");
		opts.min_nl_update = infopts.get(c_pseudotime+"."+"min_nonlinear_relaxation_factor",
		                                           0.2);
	}
	else {
		// for explicit solvers, set this any way because an inviscid flux will be constructed for the
		//  Jacobian anyway
		opts.invfluxjac = opts.invflux;
	}
	
	// check for some PETSc options
	char petsclogfile[200];
	PetscBool set = PETSC_FALSE;
	PetscOptionsGetString(NULL, NULL, "-fvens_log_file_prefix", petsclogfile, 200, &set);
	if(set)
		opts.logfile = petsclogfile;

	return opts;
}

FlowPhysicsConfig extract_spatial_physics_config(const FlowParserOptions& opts)
{
	const FlowPhysicsConfig pconf { 
		opts.gamma, opts.Minf, opts.Tinf, opts.Reinf, opts.Pr, opts.alpha,
		opts.viscsim, opts.useconstvisc, opts.bcconf
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

/* In the bc section of the control file, the various boundary conditions are given in sections
 * called 'bc0', 'bc1' etc. numbered consecutively from 0.
 * Each BC has the fields 'type' (string), 'marker' (int).
 * Some BCs have one or both of the following fields:
 * - 'boundary_values' (string of space-seperated real numbers),
 * - 'options' (string of space-separated integers).
 */
static std::vector<FlowBCConfig> parse_BC_options(const pt::ptree& infopts,
                                                  const std::string c_bcs)
{
	bool found = true;
	int ibc = 0;
	std::vector<FlowBCConfig> bcvec;

	while(found) {
		auto bc = infopts.get_optional<std::string>(c_bcs+".bc"+std::to_string(ibc));

		if(bc) {
			FlowBCConfig bconf;

			std::string bctype = boost::to_lower_copy<std::string>
				(infopts.get<std::string>(c_bcs+".bc"+std::to_string(ibc)+".type"));
			bconf.bc_type = bcTypeMap.right.at(bctype);
			bconf.bc_tag = infopts.get<int>(c_bcs+".bc"+std::to_string(ibc)+".marker");

			// Read array of boundary values
			if(bconf.bc_type == ADIABATIC_WALL_BC  ||
			   bconf.bc_type == ISOTHERMAL_WALL_BC ||
			   bconf.bc_type == SUBSONIC_INFLOW_BC)   {
				std::string bvals = infopts.get<std::string>(c_bcs+".bc"+std::to_string(ibc)+
				                                             ".boundary_values");
				bconf.bc_vals = parseStringToVector<freal>(bvals);
			}

			// array of options
			if(bconf.bc_type == PERIODIC_BC) {
				std::string bcopts = infopts.get<std::string>(c_bcs +".bc"+std::to_string(ibc)+
				                                              ".options");
				bconf.bc_opts = parseStringToVector<int>(bcopts);
			}

			bcvec.push_back(bconf);
			ibc++;
		}
		else
			found = false;
	}

	return bcvec;
}

}

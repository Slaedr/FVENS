/** \file controlparser.hpp
 * \brief Functions for parsing the main simulation control file
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef FVENS_CONTROL_PARSER_H
#define FVENS_CONTROL_PARSER_H

#include <vector>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include "aconstants.hpp"
#include "spatial/flow_spatial.hpp"

namespace fvens {

/// Data read from a control file for flow problems
struct FlowParserOptions
{
	std::string meshfile, vtu_output_file,
		logfile,                           ///< File to log timing data in
		flowtype,                          ///< Type of flow to simulate - EULER, NAVIERSTOKES
		init_soln_file,                    ///< File to read initial solution from (not implemented)
		invflux, invfluxjac,               ///< Inviscid numerical flux
		gradientmethod, limiter,           ///< Reconstruction type
		pseudotimetype,                      ///< Explicit or implicit time stepping
		constvisc,                         ///< NO for Sutherland viscosity
		surfnameprefix, volnameprefix,     ///< Filename prefixes for output files
		vol_output_reqd,                   ///< Whether volume output is required in a text file
		                                   ///<  in addition to the main VTU output
		sim_type,                          ///< Steady or unsteady simulation
		time_integrator,                   ///< Physical time discretization scheme
		/// How to compute under-relaxation factors for implicit pseudo-time or Newton solvers
		nl_update_scheme;

	freal initcfl, endcfl,                  ///< Starting CFL number and max CFL number
		tolerance,                           ///< Relative tolerance for the whole nonlinear problem
		firstinitcfl, firstendcfl,           ///< Starting and max CFL numbers for starting problem
		firsttolerance,                      ///< Relative tolerance for starting problem
		Minf, alpha, Reinf, Tinf,            ///< Free-stream flow properties
		Pr, gamma,                           ///< Non-dimensional constants Prandtl no., adia. index
		limiter_param,                       ///< Parameter controlling some limiters
		final_time,                          ///< Physical time upto which to simulate
		phy_timestep,                        ///< Constant physical time step for unsteady implicit
		phy_cfl;                             ///< CFL used only by unsteady explicit solvers
	freal min_nl_update;                    ///< Minimum under-relaxation factor for nonlinear updates

	int maxiter,
		rampstart, rampend,
		firstmaxiter,
		firstrampstart, firstrampend,
		num_out_walls,                    ///< Number of wall boundary markers where output is needed
		num_out_others,                   ///< Number of other boundaru markers where output is needed
		time_order;                       ///< Desired order of accuracy in time

	std::vector<FlowBCConfig> bcconf;     ///< All info about boundary conditions

	short soln_init_type,
		  usestarter;                     ///< Whether to start with a first-order solver initially

	bool lognres,                   ///< Whether to write out nonlinear residual history
		write_final_lin_sys,        ///< Whether to write out the last solved linear system to files
		useconstvisc,               ///< Whether to use constant viscosity instead of Sutherland
		viscsim,                    ///< Whether to carry out a viscous flow simulation
		order2;                     ///< Whether 2nd order in space is required

	std::vector<int> lwalls,         ///< List of wall boundary markers for output
		lothers;                     ///< List of other boundary markers for output
};

/// Reads a control file for flow problems
/** Replaces some option values read from the control file with those read from the command line
 * if the latter have been supplied by the user.
 */
FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[],
                                         const boost::program_options::variables_map cmdvars);

/// Extracts the spatial physics configuration from the parsed control file data
FlowPhysicsConfig extract_spatial_physics_config(const FlowParserOptions& opts);

/// Extracts the spatial discretization's settings from the parsed control file data
FlowNumericsConfig extract_spatial_numerics_config(const FlowParserOptions& opts);

/// Extracts some numerical settings for spatial discretization from the control options but
/// sets others as appropriate for a first-order scheme
FlowNumericsConfig firstorder_spatial_numerics_config(const FlowParserOptions& opts);

}

#endif

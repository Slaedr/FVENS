/** \file autilities.hpp
 * \brief Some helper functions for mundane tasks.
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef AUTILITIES_H
#define AUTILITIES_H

#include <fstream>
#include "aconstants.hpp"
#include "aspatial.hpp"
#include "aodesolver.hpp"

namespace acfd {

/// Opens a file for reading but aborts in case of an error
void open_file_toRead(const std::string file, std::ifstream& fin);

/// Opens a file for writing but aborts in case of an error
void open_file_toWrite(const std::string file, std::ofstream& fout);

/// Data read from a control file for flow problems
struct FlowParserOptions
{
	std::string meshfile, vtu_output_file, 
		logfile,                           ///< File to log timing data in
		simtype,                           ///< Type of flow to simulate - EULER, NAVIERSTOKES
		init_soln_file,                    ///< File to read initial solution from (not implemented)
		invflux, invfluxjac,               ///< Inviscid numerical flux
		gradientmethod, limiter,           ///< Reconstruction type
		timesteptype,                      ///< Explicit or implicit time stepping
		constvisc,                         ///< NO for Sutherland viscosity
		surfnameprefix, volnameprefix,     ///< Filename prefixes for output files
		vol_output_reqd;                   ///< Whether volume output is required in a text file
		                                   ///<  in addition to the main VTU output
	
	a_real initcfl, endcfl,                     ///< Starting CFL number and max CFL number
		tolerance,                              ///< Relative tolerance for the whole nonlinear problem
		firstinitcfl, firstendcfl,              ///< Starting and max CFL numbers for starting problem
		firsttolerance,                         ///< Relative tolerance for starting problem
		Minf, alpha, Reinf, Tinf,               ///< Free-stream flow properties
		Pr, gamma,                              ///< Non-dimensional constants Prandtl no., adia. index
		twalltemp, twallvel,                    ///< Isothermal wall temperature and tang. velocity
		adiawallvel,                            ///< Adiabatic wall tangential velocity magnitude
		tpwalltemp, tpwallpressure, tpwallvel;  ///< Deprecated
	
	int maxiter, 
		rampstart, rampend, 
		firstmaxiter, 
		firstrampstart, firstrampend,
		farfield_marker, inout_marker, slipwall_marker, isothermalwall_marker,
		isothermalpressurewall_marker, adiabaticwall_marker, 
		extrap_marker, 
		periodic_marker, 
		periodic_axis, 
		num_out_walls, 
		num_out_others;
	
	short soln_init_type, 
		  usestarter;
	
	bool lognres, 
		 useconstvisc, 
		 viscsim,
		 order2;
	
	std::vector<int> lwalls, 
		lothers;
};

/// Reads a control file for flow problems
const FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[]);

/// Extracts the spatial physics configuration from the parsed control file data
FlowPhysicsConfig extract_spatial_physics_config(const FlowParserOptions& opts);

/// Extracts the spatial discretization's settings from the parsed control file data
FlowNumericsConfig extract_spatial_numerics_config(const FlowParserOptions& opts);

}

#endif

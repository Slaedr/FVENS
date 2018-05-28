/** \file aoptionparser.hpp
 * \brief Some helper functions for parsing options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef AOPTIONPARSER_H
#define AOPTIONPARSER_H

#include <fstream>
#include "aconstants.hpp"
#include "spatial/aspatial.hpp"
#include "ode/aodesolver.hpp"

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
		flowtype,                          ///< Type of flow to simulate - EULER, NAVIERSTOKES
		init_soln_file,                    ///< File to read initial solution from (not implemented)
		invflux, invfluxjac,               ///< Inviscid numerical flux
		gradientmethod, limiter,           ///< Reconstruction type
		timesteptype,                      ///< Explicit or implicit time stepping
		constvisc,                         ///< NO for Sutherland viscosity
		surfnameprefix, volnameprefix,     ///< Filename prefixes for output files
		vol_output_reqd,                   ///< Whether volume output is required in a text file
		                                   ///<  in addition to the main VTU output
		sim_type,                          ///< Steady or unsteady simulation
		time_integrator;                   ///< Physical time discretization scheme
	
	a_real initcfl, endcfl,                  ///< Starting CFL number and max CFL number
		tolerance,                           ///< Relative tolerance for the whole nonlinear problem
		firstinitcfl, firstendcfl,           ///< Starting and max CFL numbers for starting problem
		firsttolerance,                      ///< Relative tolerance for starting problem
		Minf, alpha, Reinf, Tinf,            ///< Free-stream flow properties
		Pr, gamma,                           ///< Non-dimensional constants Prandtl no., adia. index
		limiter_param,                       ///< Parameter controlling some limiters
		twalltemp, twallvel,                 ///< Isothermal wall temperature and tang. velocity
		adiawallvel,                         ///< Adiabatic wall tangential velocity magnitude
		tpwalltemp, tpwallpressure, tpwallvel,  ///< Deprecated
		final_time,                          ///< Physical time upto which to simulate
		phy_timestep,                        ///< Constant physical time step for unsteady implicit
		phy_cfl;                             ///< CFL used only by unsteady explicit solvers
	
	int maxiter, 
		rampstart, rampend, 
		firstmaxiter, 
		firstrampstart, firstrampend,
		farfield_marker, inout_marker, 
		slipwall_marker, isothermalwall_marker,
		adiabaticwall_marker,  
		isothermalpressurewall_marker,          ///< Deprecated
		extrap_marker, 
		periodic_marker, 
		periodic_axis, 
		num_out_walls,                    ///< Number of wall boundary markers where output is needed
		num_out_others,                   ///< Number of other boundaru markers where output is needed
		time_order;                       ///< Desired order of accuracy in time
	
	short soln_init_type, 
		  usestarter;                     ///< Whether to start with a first-order solver initially
	
	bool lognres, 
		 useconstvisc, 
		 viscsim,
		 order2;                     ///< Whether 2nd order in space is required
	
	std::vector<int> lwalls,         ///< List of wall boundary markers for output
		lothers;                     ///< List of other boundary markers for output
};

/// Reads a control file for flow problems
const FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[]);

/// Extracts the spatial physics configuration from the parsed control file data
FlowPhysicsConfig extract_spatial_physics_config(const FlowParserOptions& opts);

/// Extracts the spatial discretization's settings from the parsed control file data
FlowNumericsConfig extract_spatial_numerics_config(const FlowParserOptions& opts);

/// Extracts some numerical settings for spatial discretization from the control options but
/// sets others as appropriate for a first-order scheme
FlowNumericsConfig firstorder_spatial_numerics_config(const FlowParserOptions& opts);

/// Extracts an integer corresponding to the argument from the default PETSc options database 
/** Throws an exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 */
int parsePetscCmd_int(const std::string optionname);

/// Optionally extracts a real corresponding to the argument from the default PETSc options database 
/** Throws an exception if the function to read the option fails, but not if it succeeds and reports
 * that the option was not set.
 * \param optionname Name of the option to be extracted
 * \param defval The default value to be assigned in case the option was not passed
 */
PetscReal parseOptionalPetscCmd_real(const std::string optionname, const PetscReal defval);

/// Extracts a boolean corresponding to the argument from the default PETSc options database 
/** Throws an exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 */
bool parsePetscCmd_bool(const std::string optionname);

/// Extracts a string corresponding to the argument from the default PETSc options database 
/** Throws a string exception if the option was not set or if it could not be extracted.
 * \param optionname The name of the option to get the value of; needs to include the preceding '-'
 * \param len The max number of characters expected in the string value
 */
std::string parsePetscCmd_string(const std::string optionname, const size_t len);

/// Extracts the arguments of an int array option from the default PETSc options database
/** \param maxlen Maximum number of entries expected in the array
 * \return The vector of array entries; its size is the number of elements read, no more
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen);

/// Extracts the arguments of an int array option from the default PETSc options database
/** Does not throw if the requested option was not found; just returns an empty vector in that case. 
 * \param maxlen Maximum number of entries expected in the array
 */
std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen);

}

#endif

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
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen);

/// Extracts the arguments of an int array option from the default PETSc options database
/** Does not throw if the requested option was not found; just returns an empty vector in that case. 
 * \param maxlen Maximum number of entries expected in the array
 */
std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen);

/// An exception to throw for errors from PETSc; takes a custom message
class Petsc_exception : public std::runtime_error
{
public:
	Petsc_exception(const std::string& msg);
	Petsc_exception(const char *const msg);
};

/// Throw an error from an error code
/** \param ierr An expression which, if true, triggers the exception
 * \param str A short string message describing the error
 */
inline void fvens_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw std::runtime_error(str);
}

/// Throw an error from an error code related to PETSc
/** \param ierr An expression which, if true, triggers the exception
 * \param str A short string message describing the error
 */
inline void petsc_throw(const int ierr, const std::string str) {
	if(ierr != 0) 
		throw acfd::Petsc_exception(str);
}

}

#endif

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
	std::string meshfile, vtu_output_file, logfile, simtype, init_soln_file,
		 invflux, invfluxjac, gradientmethod, limiter,
		 linsolver, preconditioner, timesteptype,
		 constvisc, surfnameprefix, volnameprefix, vol_output_reqd;
	
	a_real initcfl, endcfl, tolerance, lintol, 
		firstinitcfl, firstendcfl, firsttolerance,
		Minf, alpha, Reinf, Tinf, Pr, gamma, twalltemp, twallvel,
		tpwalltemp, tpwallpressure, tpwallvel, adiawallvel;
	
	int maxiter, linmaxiterstart, linmaxiterend, rampstart, rampend, 
		firstmaxiter, firstrampstart, firstrampend,
		restart_vecs, farfield_marker, inout_marker, slipwall_marker, isothermalwall_marker,
		isothermalpressurewall_marker, adiabaticwall_marker, 
		extrap_marker, periodic_marker, periodic_axis, num_out_walls, num_out_others;
	
	short soln_init_type, 
		  usestarter;
	
	unsigned short nbuildsweeps, 
				   napplysweeps;
	
	bool use_matrix_free, 
		 lognres, 
		 useconstvisc, 
		 viscsim,
		 order2, 
		 residualsmoothing;
	
	char mattype;
	
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

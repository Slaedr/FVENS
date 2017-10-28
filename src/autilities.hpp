/** \file autilities.hpp
 * \brief Some helper functions for mundane tasks.
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef AUTILITIES_H
#define AUTILITIES_H

#include <fstream>
#include "aconstants.hpp"

namespace acfd {

/// Opens a file for reading but aborts in case of an error
std::ifstream open_file_toRead(const std::string file);

/// Opens a file for writing but aborts in case of an error
std::ofstream open_file_toWrite(const std::string file);

/// Data read from a control file for flow problems
struct FlowParserOptions
{
	std::string meshfile, vtu_output_file, logfile, simtype, init_soln_file;
	std::string invflux, invfluxjac, gradientmethod, limiter;
	std::string linsolver, preconditioner, timesteptype;
	std::string constvisc, surfnameprefix, volnameprefix, vol_output_reqd;
	
	a_real initcfl, endcfl, tolerance, lintol, 
		   firstinitcfl, firstendcfl, firsttolerance;
	a_real Minf, alpha, Reinf, Tinf, Pr, gamma, twalltemp, twallvel;
	a_real tpwalltemp, tpwallpressure, tpwallvel, adiawallvel;
	
	int maxiter, linmaxiterstart, linmaxiterend, rampstart, rampend, 
		firstmaxiter, firstrampstart, firstrampend,
		restart_vecs, farfield_marker, inout_marker, slipwall_marker, isothermalwall_marker,
		isothermalpressurewall_marker, adiabaticwall_marker, 
		extrap_marker, periodic_marker, periodic_axis;
	
	int num_out_walls, num_out_others;
	
	short soln_init_type, usestarter;
	
	unsigned short nbuildsweeps, napplysweeps;
	
	bool use_matrix_free=false, lognres, useconstvisc=false, viscsim=false,
		 order2 = true, residualsmoothing = false;
	
	char mattype;
	
	std::vector<int> lwalls, lothers;
};

/// Reads a control file for flow problems
const FlowParserOptions parse_flow_controlfile(const int argc, const char *const argv[]);

}

#endif

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
	
	a_real initcfl, endcfl, 
		tolerance, 
		firstinitcfl, firstendcfl, 
		firsttolerance,
		Minf, alpha, Reinf, Tinf, 
		Pr, gamma, 
		twalltemp, twallvel,
		adiawallvel,
		tpwalltemp, tpwallpressure, tpwallvel;
	
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

/// Computes various entity lists required for mesh traversal, also reorders the cells if requested
/** Does not compute periodic boundary maps; this must be done separately. 
 * \ref UMesh2dh::compute_periodic_map
 */
StatusCode preprocessMesh(UMesh2dh& m);

/// Reorders the mesh cells in a given ordering using PETSc
/** Symmetric premutations only.
 * \warning It is the caller's responsibility to recompute things that are affected by the reordering,
 * such as \ref UMesh2dh::compute_topological.
 *
 * \param ordering The ordering to use - "rcm" is recommended. See the relevant
 * [page](www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatOrderingType.html)
 * in the PETSc manual for the full list.
 * \param sd Spatial discretization to be used to generate a Jacobian matrix
 * \param m The mesh context
 */
StatusCode reorderMesh(const char *const ordering, const Spatial<1>& sd, UMesh2dh& m);

}

#endif

/** \file casesolvers.hpp
 * \brief Routines to solve a single fluid dynamics case
 * \author Aditya Kashi
 * \date 2018-04
 */

#ifndef FVENS_CASESOLVERS_H
#define FVENS_CASESOLVERS_H

#include <string>
#include <petscvec.h>
#include "utilities/aoptionparser.hpp"

namespace acfd {

/// Integrated quantities of interest in the solution of a flow problem
struct FlowSolutionFunctionals
{
	a_real meshSizeParameter;   ///< Not a solution functional but needed for verification studies
	a_real entropy;             ///< Any measure of entropy difference from the free-stream
	a_real CL;                  ///< Lift coefficient
	a_real CDp;                 ///< Coefficient of drag induced by pressure
	a_real CDsf;                ///< Coefficient of drag caused by skin-friction
};

/// Solves a steady-state case on one mesh
/** Writes no output files. \sa steadyCase_output
 *
 * \param opts Parsed options from a control file
 * \param mesh_suffix A string to concatenate to the [mesh file name](\ref FlowParserOptions::meshfile)
 *   before passing to the mesh class to be read.
 * \param u Pointer to an uninitialized PETSc vec used for storing the solution; must be destroyed
 *   explicitly by the caller after it has been used.
 */
int steadyCase(const FlowParserOptions& opts, const std::string mesh_suffix, Vec *const u);

/// Solves a steady-state case on one mesh
/** Writes ASCII surface and volume output files depending on the control database. May also write
 * a VTU file of the volume output.
 *
 * \param opts Parsed options from a control file
 * \param mesh_suffix A string to concatenate to the [mesh file name](\ref FlowParserOptions::meshfile)
 *   before passing to the mesh class to be read.
 * \param u Pointer to an uninitialized PETSc vec used for storing the solution; must be destroyed
 *   explicitly by the caller after it has been used.
 * \param vtu_output_needed If true, writes out a VTU output file to a filename determined by the
 *   control database.
 */
FlowSolutionFunctionals steadyCase_output(const FlowParserOptions& opts, const std::string mesh_suffix, 
	Vec *const u, const bool vtu_output_needed);

}
#endif

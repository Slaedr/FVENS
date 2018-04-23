/** \file casesolvers.hpp
 * \brief Routines to solve a single fluid dynamics case
 * \author Aditya Kashi
 * \date 2018-04
 */

#ifndef FVENS_CASESOLVERS_H
#define FVENS_CASESOLVERS_H

#include <string>
#include <tuple>
#include <petscvec.h>
#include "utilities/autilities.hpp"

namespace acfd {

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
std::tuple<a_real,a_real,a_real> 
steadyCase_output(const FlowParserOptions& opts, const std::string mesh_suffix, Vec *const u,
	const bool vtu_output_needed);

}
#endif

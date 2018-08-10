/** \file casesolvers.hpp
 * \brief Routines to solve a single fluid dynamics case
 * \author Aditya Kashi
 * \date 2018-04
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

#ifndef FVENS_CASESOLVERS_H
#define FVENS_CASESOLVERS_H

#include <string>
#include <petscvec.h>
#include "utilities/controlparser.hpp"
#include "utilities/aoptionparser.hpp"

namespace fvens {

/// Integrated quantities of interest in the solution of a flow problem
struct FlowSolutionFunctionals
{
	a_real meshSizeParameter;   ///< Not a solution functional but needed for verification studies
	a_real entropy;             ///< Any measure of entropy difference from the free-stream
	a_real CL;                  ///< Lift coefficient
	a_real CDp;                 ///< Coefficient of drag induced by pressure
	a_real CDsf;                ///< Coefficient of drag caused by skin-friction
};

class FlowCase
{
public:
	/// Construct a flow case with parsed options
	/** \param opts Parsed options from a control file
	 */
	FlowCase(const FlowParserOptions& options);

	/// Setup and run a case without writing output files and without computing output functionals
	/** \param mesh_suffix A string to concatenate to the
	 *   [mesh file name](\ref FlowParserOptions::meshfile) before passing to the mesh class.
	 * \param u Pointer to an uninitialized PETSc vec used for storing the solution;
	 *   must be destroyed explicitly by the caller after it has been used.
	 */
	virtual int run(const std::string mesh_suffix, Vec *const u) const;

	/// Setup and run a case, return some functionals of interest and optionally write output files
	/** \param mesh_suffix A string to concatenate to the
	 *   [mesh file name](\ref FlowParserOptions::meshfile) before passing to the mesh class.
	 * \param vtu_output_needed Whether writing the full solution to VTU files is desired.
	 * \param u Pointer to an uninitialized PETSc vec used for storing the solution;
	 *   must be destroyed explicitly by the caller after it has been used.
	 */
	virtual FlowSolutionFunctionals run_output(const std::string mesh_suffix,
												const bool vtu_output_needed,
												Vec *const u) const;

	/// Construct a mesh from the base mesh name in the [options database](\ref FlowParserOptions)
	///  and a suffix
	/** \param mesh_suffix A string to concatenate to the
	 *   [mesh file name](\ref FlowParserOptions::meshfile) before passing to the mesh class.
	 */
	UMesh2dh<a_real> constructMesh(const std::string mesh_suffix) const;

	/// Solve a case given a spatial discretization context
	/** Specific case types must provide an implementation of this.
	 * \return An error code (may also throw exceptions)
	 */
	virtual int execute(const Spatial<a_real,NVARS> *const prob, Vec u) const = 0;

protected:
	const FlowParserOptions& opts;
};

/// Solution procedure for a steady-state case
/** To use, one should just call either \ref FlowCase::run or \ref FlowCase::run_output.
 * This class just provides the underlying implementation.
 */
class SteadyFlowCase : public FlowCase
{
public:
	/// Construct a flow case with parsed options
	SteadyFlowCase(const FlowParserOptions& options);

	/// Solve a case given a spatial discretization context
	int execute(const Spatial<a_real,NVARS> *const prob, Vec u) const;
};

/// Solution procedure for an unsteady flow case
/** To use, one should just call either \ref FlowCase::run or \ref FlowCase::run_output.
 * Currently only TVD RK time integration is supported.
 */
class UnsteadyFlowCase : public FlowCase
{
public:
	/// Construct a flow case with parsed options
	UnsteadyFlowCase(const FlowParserOptions& options);

	/// Solve a case given a spatial discretization context
	int execute(const Spatial<a_real,NVARS> *const prob, Vec u) const;
};

}
#endif

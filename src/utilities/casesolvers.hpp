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
#include <petscksp.h>
#include "linalg/alinalg.hpp"
#include "ode/aodesolver.hpp"
#include "utilities/controlparser.hpp"
#include "utilities/aerrorhandling.hpp"

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

/// Construct a mesh from the base mesh name in the [options database](\ref FlowParserOptions)
///  and a suffix
/** Reads the mesh, computes the face connectivity, reorders the cells if requested and
 * sets up periodic boundaries if necessary.
 * \param mesh_suffix A string to concatenate to the
 *   [mesh file name](\ref FlowParserOptions::meshfile) before passing to the mesh class.
 */
UMesh2dh<a_real> constructMeshFlow(const FlowParserOptions& opts, const std::string mesh_suffix);

/// Create a spatial discretization context for the flow problem
const FlowFV_base<a_real>* createFlowSpatial(const FlowParserOptions& opts,
                                             const UMesh2dh<a_real>& m);

/// Allocate a vector of size number of cells times the number of PDEs, and initialize it with
///  free-stream values from control file options
int initializeSystemVector(const FlowParserOptions& opts, const UMesh2dh<a_real>& m, Vec *const u);

/// Solve a flow problem, either steady or unsteady, with conditions specified in the FVENS control file
/** \todo Ideally, the solution vector would be owned by the nonlinear solver to accommodate adaptation,
 * There would be a mechanism to return the final solution vector from the ODE solver, through the
 * case solver and to the caller of the flow case solve.
 */
class FlowCase
{
public:
	/// Construct a flow case with parsed options
	/** \param opts Parsed options from a control file
	 */
	FlowCase(const FlowParserOptions& options);

	/// Setup and run a case without writing output files and without computing output functionals
	/** \param mesh The mesh context
	 * \param u An allocated and initialized PETSc vec used for storing the solution
	 */
	virtual int run(const UMesh2dh<a_real>& mesh, Vec u) const;

	/// Setup and run a case, return some functionals of interest and optionally write output files
	/** Whether VTU volume output and surface variable output is required to files is given by
	 * arguments here. Whether volume variable output (non-VTU) is required is taken from the
	 * options database \ref FlowParserOptions::vol_output_reqd .
	 * 
	 * \param[in] surface_file_needed True if the solution on relevant surfaces should be written
	 *   out to files, else set to false
	 * \param[in] vtu_output_needed Whether writing the full solution to VTU files is desired.
	 * \param[in] mesh The mesh to solve the problem on
	 * \param[in,out] u Pointer to an allocated and initialized PETSc vec used for storing the solution
	 */
	virtual FlowSolutionFunctionals run_output(const bool surface_file_needed,
	                                           const bool vtu_output_needed,
	                                           const UMesh2dh<a_real>& mesh, Vec u) const;

	/// Solve a case given a spatial discretization context
	/** Specific case types must provide an implementation of this.
	 * \param output_conv_history If true, residual history and preconditioner timing are written to
	 *   (two separate) files
	 * \return An error code (may also throw exceptions)
	 */
	virtual int execute(const Spatial<a_real,NVARS> *const prob, const bool output_conv_history,
	                    Vec u) const = 0;

	/// Solve a startup problem corresponding to the actual problem to be solved
	/**
	 * Should set up all the implicit solver objects it needs and destroy them after it's done.
	 * \param[in] prob The original problem to be solved
	 * \param[in,out] u Solution vector - contains initial condition on input and solution on output
	 */
	virtual int execute_starter(const Spatial<a_real,NVARS> *const prob, Vec u) const = 0;

	/// Solve the problem from some (decent) initial condition
	/**
	 * Should set up all the implicit solver objects it needs and destroy them after it's done.
	 * \param[in] prob The problem to be solved
	 * \param[in,out] u Solution vector - contains initial condition on input and solution on output
	 * \return Time taken by various phases of the solve and whether or not it converged
	 */
	virtual TimingData execute_main(const Spatial<a_real,NVARS> *const prob, Vec u) const = 0;

protected:
	const FlowParserOptions& opts;

	/// Objects required for time-implicit solution
	struct LinearProblemLHS {
		Mat A;                                  ///< System Jacobian matrix
		Mat M;                                  ///< System preconditioning matrix
		KSP ksp;                                ///< Linear solver context
		bool mf_flg;                            ///< Whether matrix-free Jacobian has been requested

		/// Destroy all components of linear problem LHS
		int destroy() {
			int ierr = 0;
			ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
			ierr = MatDestroy(&M); CHKERRQ(ierr);
			if(mf_flg) {
				ierr = MatDestroy(&A); CHKERRQ(ierr);
			}
			return ierr;
		}
	};

	/// Sets up matrices and KSP contexts
	/** Sets KSP options from command line (or PETSc options file) as well.
	 * \param[in] s The spatial discretization for which to set up the solver
	 * \param[in] use_mfjac Whether a matrix-free Jacobian should be set up (true) or not (false)
	 * \return Objects required for implicit solution of the problem
	 */
	static LinearProblemLHS setupImplicitSolver(const Spatial<a_real,NVARS> *const s, const bool use_mfjac);

	/// Sets up only the KSP context, assuming the Mats have been set up
	static void setupKSP(LinearProblemLHS& solver, const bool use_matrix_free);
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
	/** Throws a \ref Tolerance_error if the main solve does not converge.
	 * \param output_reshistory If true, residual history is written to file
	 */
	int execute(const Spatial<a_real,NVARS> *const prob, const bool output_reshistory, Vec u) const;

	/// Solve the 1st-order problem corresponding to the actual problem to be solved
	/** Even if the solver does not converge to required tolerance, no exception is thrown.
	 * Sets up all the implicit solver objects it needs and destroys them after it's done.
	 * \param[in] prob The original problem to be solved
	 * \param[in,out] u Solution vector - contains initial condition on input and solution on output
	 */
	int execute_starter(const Spatial<a_real,NVARS> *const prob, Vec u) const;

	/// Solve the steady-state problem from some (decent) initial condition
	/** If the solver does not converge to required tolerance, a \ref Tolerance_error is thrown.
	 * Sets up all the implicit solver objects it needs and destroys them after it's done.
	 * \param[in] prob The problem to be solved
	 * \param[in,out] u Solution vector - contains initial condition on input and solution on output
	 * \return Time taken by various phases of the solve and whether or not it converged
	 */
	TimingData execute_main(const Spatial<a_real,NVARS> *const prob, Vec u) const;

protected:

	const bool mf_flg;
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

/** @file alinalg.hpp
 * @brief Setup and handling of some linear algebra objects 
 * @author Aditya Kashi
 */

#ifndef ALINALG_H
#define ALINALG_H

#include "aconstants.hpp"
#include "amesh2dh.hpp"
#include "aspatial.hpp"

#include <petscksp.h>

#define PETSCOPTION_STR_LEN 30

#ifdef USE_BLASTED
#include <blasted_petsc.h>
#endif

namespace acfd {

/// Sets up storage preallocation for sparse matrix formats
/** \param[in] m Mesh context
 * \param[in|out] A The matrix to pre-allocate for
 *
 * We assume there's only 1 neighboring cell that's not in this subdomain
 * \todo TODO: Once a partitioned mesh is used, set the preallocation properly.
 *
 * The type of sparse matrix is read from the PETSc options database, but
 * only MPI matrices are supported.
 */
template <int nvars>
StatusCode setupSystemMatrix(const UMesh2dh *const m, Mat *const A);

/// Computes the amount of memory to be reserved for the Jacobian matrix
template <int nvars>
StatusCode setJacobianPreallocation(const UMesh2dh *const m, Mat A);

/// Matrix-free Jacobian of the flux
/** The normalized step length epsilon for the finite-difference Jacobian is set to a default value,
 * but it also queried from the PETSc options database.
 */
template <int nvars>
class MatrixFreeSpatialJacobian
{
public:
	/// Query the Petsc options database for a custom option for setting the step length epsilon
	/** The finite difference step length epsilon is given a default value.
	 */
	MatrixFreeSpatialJacobian();

	/// Set the spatial dscretization whose Jacobian is needed
	void set_spatial(const Spatial<nvars> *const space);

	/// Allocate storage for work vectors using the (possibly matrix-free) Mat as a template
	StatusCode setup_work_storage(const Mat system_matrix);

	/// Release storage from work vectors
	StatusCode destroy_work_storage();

	/// Set the state u at which the Jacobian is computed, the corresponding residual r(u) and 
	/// the diagonal vector of the mass matrix for each cell
	/** Note that the residual vector supplied is assumed to be the negative of what is needed,
	 * exactly what Spatial::compute_residual gives.
	 */
	void set_state(const Vec u_state, const Vec r_state, const std::vector<a_real> *const mdts);

	/// Compute a Jacobian-vector product
	StatusCode apply(const Vec x, Vec y) const;

protected:
	/// Spatial discretization context
	const Spatial<nvars>* spatial;

	/// step length for finite difference Jacobian
	a_real eps;

	/// The state at which to compute the Jacobian
	Vec u;

	/// The residual of the state \ref uvec at which to compute the Jacobian
	Vec res;

	/// Time steps for each cell
	const std::vector<a_real> *mdt;

	/// Temporary storage
	mutable Vec aux;
};

/// Setup a matrix-free Mat for the Jacobian
/**
 * \param mfj A constructed MatrixFreeSpatialJacobian object
 * \param A The Mat to setup We assume \ref setupSystemMatrix has already been called on the Mat
 *   to set the size etc.
 */
template <int nvars>
StatusCode setup_matrixfree_jacobian(const UMesh2dh *const m,
		MatrixFreeSpatialJacobian<nvars> *const mfj, Mat *const A);

/// Returns true iff the argument is a matrix-free PETSc Mat
bool isMatrixFree(Mat);

#ifdef USE_BLASTED

/// Sets BLASTed preconditioners
/** Only one subdomain per rank is supported in case of domain decomposition global preconditioners.
 * \param ksp The top-level KSP
 * \param u A solution vector used to assemble the Jacobian matrix once, needed for initialization
 *   of some PETSc preconditioners - the actual values don't matter.
 * \param startprob A spatial discretization context to compute the Jacobian with
 * \param[in,out] bctx The BLASTed context
 */
template <int nvars>
StatusCode setup_blasted(KSP ksp, Vec u, const Spatial<nvars> *const startprob, Blasted_data_vec& bctx);

#endif

}
#endif

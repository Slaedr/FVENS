/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef ALINALG_H
#define ALINALG_H

#include "aconstants.hpp"
#include "amesh2dh.hpp"
#include "aspatial.hpp"

#include <petscmat.h>

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

	/// Set the state u at which the Jacobian is computed and the corresponding residual r(u)
	void set_state(const Vec u_state, const Vec r_state);

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
StatusCode setup_matrixfree_jacobian(MatrixFreeSpatialJacobian<nvars> *const mfj, Mat A);

/// Returns true iff the argument is a matrix-free PETSc Mat, ie., a Mat shell
inline bool isMatrixFree(Mat M) {
	MatType mattype;
	MatGetType(M, &mattype);
	if(!strcmp(mattype,"shell"))
		return true;
	else
		return false;
}

}
#endif

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

template <int nvars>
class MatrixFreeSpatialJacobian
{
public:
	/// Set up a matrix-free Jacobian from a spatial discretization context
	/// and the finite difference step
	MatrixFreeSpatialJacobian(const Spatial<nvars> *const space, const a_real epsilon,
			const Vec system_vector);

	~MatrixFreeSpatialJacobian();

	/// Compute a Jacobian-vector product
	StatusCode apply(const Vec x, Vec y) const;

protected:
	/// Spatial discretization context
	const Spatial<nvars>* spatial;

	/// step length for finite difference Jacobian
	const a_real eps;

	/// Temporary storage
	mutable Vec aux;
};

}
#endif

/** @file aexplicitsolver.hpp
 * @brief Explicit solution of ODEs resulting from some spatial discretization
 * @author Aditya Kashi
 * @date 24 Feb 2016; modified 13 May 2017
 */
#ifndef __AEXPLICITSOLVER_H
#define __AEXPLICITSOLVER_H 1

#ifndef __ASPATIAL_H
#include "aspatial.hpp"
#endif

namespace acfd {

/// A driver class to control the explicit time-stepping solution using forward Euler time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class ForwardEulerTimeSolver
{
	const UMesh2dh *const m;				///< Mesh
	EulerFV *const eul;						///< Spatial discretization context

public:
	ForwardEulerTimeSolver(const UMesh2dh *const mesh, EulerFV *const euler);
	~ForwardEulerTimeSolver();

	/// Solves a steady problem by an explicit method first order in time, using local time-stepping
	void solve(const a_real tol, const int maxiter, const a_real cfl);

	/// Computes the L2 norm of a cell-centered quantity
	a_real l2norm(const amat::Matrix<a_real>* const v);
};

}	// end namespace
#endif

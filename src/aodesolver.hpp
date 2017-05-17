/** @file aodesolver.hpp
 * @brief Solution of ODEs resulting from some spatial discretization
 * @author Aditya Kashi
 * @date 24 Feb 2016; modified 13 May 2017
 */
#ifndef __AODESOLVER_H
#define __AODESOLVER_H 1

#ifndef __ASPATIAL_H
#include "aspatial.hpp"
#endif

#ifndef __ALINALG_H
#include "alinalg.hpp"
#endif

namespace acfd {

/// A driver class for explicit time-stepping solution to steady state using forward Euler time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class SteadyForwardEulerSolver
{
	const UMesh2dh *const m;				///< Mesh
	EulerFV *const eul;						///< Spatial discretization context

public:
	SteadyForwardEulerSolver(const UMesh2dh *const mesh, EulerFV *const euler);
	~SteadyForwardEulerSolver();

	/// Solves a steady problem by an explicit method first order in time, using local time-stepping
	void solve(const a_real tol, const int maxiter, const a_real cfl);

	/// Computes the L2 norm of a cell-centered quantity
	a_real l2norm(const amat::Array2d<a_real>* const v);
};

/// Implicit pseudo-time iteration to steady state
class SteadyBackwardEulerSolver
{
	const UMesh2dh *const m;
	
	EulerFV *const eul;

	IterativeBlockSolver * linsolv;
	Matrix* D;
	Matrix* L;
	Matrix* U;

	double cflinit;
	double cflfin;
	int rampstart;
	int rampend;
	double tol;
	int maxiter;
	int lintol;
	int linmaxiterstart;
	int linmaxiterend;

public:
	SteadyBackwardEulerSolver(const UMesh2dh*const mesh, EulerFV *const spatial,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const int lin_tol, const int linmaxiterstart, const int linmaxiterend, std::string linearsolver);
	
	~SteadyBackwardEulerSolver();

	void solve();
};

}	// end namespace
#endif

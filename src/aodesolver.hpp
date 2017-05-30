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

class SteadySolver
{
protected:
	const UMesh2dh *const m;
	EulerFV *const eul;
	EulerFV *const starter;
	Matrix residual;
	Matrix u;
	const short usestarter;

public:
	SteadySolver(const UMesh2dh *const mesh, EulerFV *const euler, EulerFV *const starterfv, const short use_starter)
		: m(mesh), eul(euler), starter(starterfv), usestarter(use_starter)
	{ }

	const Matrix& residuals() const {
		return residual;
	}
	
	/// Write access to the conserved variables
	Matrix& unknowns() {
		return u;
	}

	virtual void solve() = 0;

	virtual ~SteadySolver() {}
};
	
/// A driver class for explicit time-stepping solution to steady state using forward Euler time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_areas()
 * have been called on the mesh object prior to initialzing an object of this class.
 */
class SteadyForwardEulerSolver : public SteadySolver
{
	amat::Array2d<a_real> dtm;				///< Stores allowable local time step for each cell
	const double tol;
	const int maxiter;
	const double cfl;
	
	const double starttol;
	const int startmaxiter;
	const double startcfl;

public:
	SteadyForwardEulerSolver(const UMesh2dh *const mesh, EulerFV *const euler, EulerFV *const starterfv, 
			const short use_starter, const double toler, const int maxits, const double cfl,
			const double ftoler, const int fmaxits, const double fcfl);
	~SteadyForwardEulerSolver();

	/// Solves a steady problem by an explicit method first order in time, using local time-stepping
	void solve();
};

/// Implicit pseudo-time iteration to steady state
class SteadyBackwardEulerSolver : public SteadySolver
{
	amat::Array2d<a_real> dtm;				///< Stores allowable local time step for each cell

	IterativeBlockSolver * linsolv;
	Matrixb* D;
	Matrixb* L;
	Matrixb* U;

	const double cflinit;
	double cflfin;
	int rampstart;
	int rampend;
	double tol;
	int maxiter;
	int lintol;
	int linmaxiterstart;
	int linmaxiterend;
	
	const double starttol;
	const int startmaxiter;
	const double startcfl;

public:
	SteadyBackwardEulerSolver(const UMesh2dh*const mesh, EulerFV *const spatial, EulerFV *const starterfv, const short use_starter,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const int lin_tol, const int linmaxiterstart, const int linmaxiterend, std::string linearsolver,
		const double ftoler, const int fmaxits, const double fcfl);
	
	~SteadyBackwardEulerSolver();

	void solve();
};

}	// end namespace
#endif

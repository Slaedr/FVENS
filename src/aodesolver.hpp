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

/// Base class for steady-state simulations in pseudo-time
/** Note that the unknowns u and residuals R correspond to the following ODE:
 * \f$ \frac{du}{dt} + R(u) = 0 \f$. Note that the residual is on the LHS.
 */
template <short nvars>
class SteadySolver
{
protected:
	const UMesh2dh *const m;
	Spatial<nvars> *const eul;
	Spatial<nvars> *const starter;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> residual;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> u;
	const short usestarter;
	double cputime;
	double walltime;

public:
	SteadySolver(const UMesh2dh *const mesh, Spatial<nvars> *const euler, 
			Spatial<nvars> *const starterfv, const short use_starter)
		: m(mesh), eul(euler), starter(starterfv), usestarter(use_starter), 
			cputime{0.0}, walltime{0.0}
	{ }

	const Matrix<a_real,Dynamic,Dynamic,RowMajor>& residuals() const {
		return residual;
	}
	
	/// Write access to the conserved variables
	Matrix<a_real,Dynamic,Dynamic,RowMajor>& unknowns() {
		return u;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	virtual void solve() = 0;

	virtual ~SteadySolver() {}
};
	
/// A driver class for explicit time-stepping to steady state using forward Euler time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_areas()
 * have been called on the mesh object prior to initialzing an object of this class.
 */
template <short nvars>
class SteadyForwardEulerSolver : public SteadySolver<nvars>
{
	using SteadySolver<nvars>::m;
	using SteadySolver<nvars>::eul;
	using SteadySolver<nvars>::starter;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::u;
	using SteadySolver<nvars>::usestarter;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;

	amat::Array2d<a_real> dtm;				///< Stores allowable local time step for each cell
	const double tol;
	const int maxiter;
	const double cfl;
	
	const double starttol;
	const int startmaxiter;
	const double startcfl;

public:
	SteadyForwardEulerSolver(const UMesh2dh *const mesh, 
			Spatial<nvars> *const euler, Spatial<nvars> *const starterfv, 
			const short use_starter, const double toler, const int maxits, const double cfl,
			const double ftoler, const int fmaxits, const double fcfl);
	~SteadyForwardEulerSolver();

	/// Solves the steady problem by a first-order explicit method, using local time-stepping
	void solve();
};

/// Implicit pseudo-time iteration to steady state
template <short nvars>
class SteadyBackwardEulerSolver : public SteadySolver<nvars>
{
	using SteadySolver<nvars>::m;
	using SteadySolver<nvars>::eul;
	using SteadySolver<nvars>::starter;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::u;
	using SteadySolver<nvars>::usestarter;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;

	amat::Array2d<a_real> dtm;					///< Stores allowable local time step for each cell

	IterativeBlockSolver<nvars> * linsolv;		///< Linear solver context
	DLUPreconditioner<nvars>* prec;				///< preconditioner context
	Matrix<a_real,nvars,nvars,RowMajor>* D;
	Matrix<a_real,nvars,nvars,RowMajor>* L;
	Matrix<a_real,nvars,nvars,RowMajor>* U;

	const double cflinit;
	double cflfin;
	int rampstart;
	int rampend;
	double tol;
	int maxiter;
	double lintol;
	int linmaxiterstart;
	int linmaxiterend;
	
	const double starttol;
	const int startmaxiter;
	const double startcfl;

public:
	SteadyBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial, Spatial<nvars> *const starterfv, const short use_starter,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, const double lin_tol, const int linmaxiterstart, const int linmaxiterend, std::string linearsolver, std::string precond,
		const unsigned short nbuildsweeps, const unsigned short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl);
	
	~SteadyBackwardEulerSolver();

	void solve();
};

}	// end namespace
#endif

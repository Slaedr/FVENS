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
	MVector residual;
	MVector u;
	const short usestarter;
	double cputime;
	double walltime;

public:
	SteadySolver(const UMesh2dh *const mesh, Spatial<nvars> *const euler, 
			Spatial<nvars> *const starterfv, const short use_starter)
		: m(mesh), eul(euler), starter(starterfv), usestarter(use_starter), 
			cputime{0.0}, walltime{0.0}
	{ }

	const MVector& residuals() const {
		return residual;
	}
	
	/// Write access to the conserved variables
	MVector& unknowns() {
		return u;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	virtual void solve() = 0;

	virtual ~SteadySolver() {}
};
	
/// A driver class for explicit time-stepping to steady state using forward Euler integration
/** \note Make sure compute_topological(), compute_face_data() and compute_areas()
 * have been called on the mesh object prior to initialzing an object of this class.
 * 
 * Optionally runs a `starter' time stepping loop to generate an initial solution
 * before starting the `main' loop.
 * The starter can perhaps use a first-order discretization.
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
/** Optionally runs a `starter' time stepping loop to generate an initial solution
 * before starting the `main' loop.
 * The starter can, perhaps, use a lower CFL number or use a first-order discretization.
 */
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

	amat::Array2d<a_real> dtm;               ///< Stores allowable local time step for each cell

	IterativeSolver<nvars> * linsolv;        ///< Linear solver context
	Preconditioner<nvars>* prec;             ///< preconditioner context
	LinearOperator<a_real,a_int>* A;         ///< Sparse matrix to hold the Jacobian or LHS

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
	
	/// Sets required data and sets up the sparse Jacobian storage
	/** 
	 * \param[in] mesh Mesh context
	 * \param[in] spatial Spatial discretization context
	 * \param[in] starterfv Spatial discretization used to generate an approximate solution initially
	 * \param[in] use_starter Whether to use \ref starterfv to generate an initial solution
	 * \param[in] cfl_init CFL to be used when the main solver starts
	 * \param[in] cfl_fin Final CFL number attained at \ref ramp_end iterations
	 * \param[in] ramp_start Iteration number of main solver at which 
	 *            CFL number begins to linearly increase
	 * \param[in] ramp_end Iteration number of the main solver at which CFL ramping ends
	 * \param[in] toler Relative residual tolerance for the ODE solver for the main solver
	 * \param[in] maxits Maximum number of pseudo-time steps for the main solver
	 * \param[in] mat_type A character which selects the matrix storage scheme for the Jacobian.
	 *            Possible values: 'p' (point CSR storage), 'b' (block CSR storage) 
	 *            or 'd' ('DLU' storage)
	 * \param[in] linmaxiterstart Maximum iterations per time step for the linear solver
	 *              both for the starting ODE solver and initially for the main ODE solver
	 * \param[in] linmaxiterend Maximum iterations per time step at the end of the CFL ramping
	 *              of the main ODE solver
	 * \param[in] linearsolver Selects the linear solver to use; possible values:
	 *              "RICHARDSON", "BCGSTB" (BiCGStab)
	 * \param[in] precond Selects preconditioner to use for the linear solver; possible values:
	 *              "BSGS", "BILU0", "BJ"
	 * \param[in] nbuildsweeps Number of sweeps to use while asynchronously building 
	 *            the ILU0 preconditioner
	 * \param[in] napplysweeps Number of sweeps to use for each asynchronous loop 
	 *            during application of preconditioners
	 * \param[in] ftoler Tolerance for the starting ODE solver
	 * \param[in] fmaxits Maximum iterations for the starting ODE solver
	 * \param[in] fcfl CFL number to use for starting ODE solver
	 * \param[in] restart_vecs Number of Krylov subspace vectors to store per restart iteration
	 */
	SteadyBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial, 
		Spatial<nvars> *const starterfv, const short use_starter,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const char mat_type, const double lin_tol, const int linmaxiterstart, 
		const int linmaxiterend, std::string linearsolver, std::string precond,
		const short nbuildsweeps, const short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl,
		const int restart_vecs);
	
	~SteadyBackwardEulerSolver();

	/// Runs the time-stepping loop
	void solve();
};

/// Implicit pseudo-time iteration to steady state using a martrix-free linear solver
template <short nvars>
class SteadyMFBackwardEulerSolver : public SteadySolver<nvars>
{
	using SteadySolver<nvars>::m;
	using SteadySolver<nvars>::eul;
	using SteadySolver<nvars>::starter;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::u;
	using SteadySolver<nvars>::usestarter;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;

	/// Stores allowable local time step for each cell
	amat::Array2d<a_real> dtm; 

	MFIterativeSolver<nvars> * startlinsolv;    ///< Linear solver context for starting run
	MFIterativeSolver<nvars> * linsolv;         ///< Linear solver context for main run
	Preconditioner<nvars>* prec;                ///< preconditioner context
	LinearOperator<a_real,a_int>* M;            ///< Preconditioning matrix

	/// Temporary storage needed for matrix-free derivative evaluation
	MVector aux;

	// TEMPORARY - REMOVE!
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
	SteadyMFBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial, 
		Spatial<nvars> *const starterfv, const short use_starter,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const double lin_tol, const int linmaxiterstart, const int linmaxiterend, 
		std::string linearsolver, std::string precond,
		const short nbuildsweeps, const short napplysweeps,
		const double ftoler, const int fmaxits, const double fcfl, const int restart_vecs);
	
	~SteadyMFBackwardEulerSolver();

	void solve();
};

}	// end namespace
#endif

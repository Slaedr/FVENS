/** @file aodesolver.hpp
 * @brief Solution of ODEs resulting from some spatial discretization
 * @author Aditya Kashi
 * @date 24 Feb 2016; modified 13 May 2017
 */
#ifndef AODESOLVER_H
#define AODESOLVER_H 1

#include "aspatial.hpp"

#include "alinalg.hpp"

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
	const Spatial<nvars> * eul;
	MVector& u;
	MVector residual;
	double cputime;
	double walltime;
	bool lognres;

public:
	/** 
	 * \param[in] mesh Mesh context
	 * \param[in] spatial Spatial discretization context
	 * \param[in] soln The solution vector to use and update
	 * \param[in] use_starter Whether to use \ref starterfv to generate an initial solution
	 * \param[in] log_nonlinear_residual Set to true if output of convergence history is needed
	 */
	SteadySolver(const UMesh2dh *const mesh, Spatial<nvars> *const spatial, MVector& soln,
			bool log_nonlinear_residual)
		: m(mesh), eul(spatial), u(soln), cputime{0.0}, walltime{0.0}, 
		  lognres{log_nonlinear_residual}
	{ }

	const MVector& residuals() const {
		return residual;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	virtual void solve(std::string logfile) = 0;

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
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::u;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;
	using SteadySolver<nvars>::lognres;

	amat::Array2d<a_real> dtm;				///< Stores allowable local time step for each cell
	const double tol;
	const int maxiter;
	const double cfl;

	/// Stores whether implicit Laplacian smoothing is to be used for the residual
	const bool useImplicitSmoothing;

	/// Sparse matrix for the Laplacian
	const LinearOperator<a_real,a_int> *const M;

	IterativeSolver<nvars> * linsolv;        ///< Linear solver context for Laplacian smoothing
	Preconditioner<nvars>* prec;             ///< preconditioner context for Laplacian smoothing

public:
	SteadyForwardEulerSolver(const UMesh2dh *const mesh, Spatial<nvars> *const euler, MVector& sol,
			const double toler, const int maxits, const double cfl,
			const bool use_implicitSmoothing, const LinearOperator<a_real,a_int> *const A,
			bool log_nonlinear_res);
	
	~SteadyForwardEulerSolver();

	/// Solves the steady problem by a first-order explicit method, using local time-stepping
	void solve(std::string logfile);
};

/// Implicit pseudo-time iteration to steady state
template <short nvars>
class SteadyBackwardEulerSolver : public SteadySolver<nvars>
{
	using SteadySolver<nvars>::m;
	using SteadySolver<nvars>::eul;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::u;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;
	using SteadySolver<nvars>::lognres;

	amat::Array2d<a_real> dtm;               ///< Stores allowable local time step for each cell

	IterativeSolver<nvars> * linsolv;        ///< Linear solver context
	Preconditioner<nvars>* prec;             ///< preconditioner context

	/// Sparse matrix of the preconditioning Jacobian
	/** Note that the same matrix is used as the actual LHS as well,
	 * if matrix-free solution is disabled.
	 */
	LinearOperator<a_real,a_int>* M;

	const double cflinit;
	double cflfin;
	int rampstart;
	int rampend;
	double tol;
	int maxiter;
	double lintol;
	int linmaxiterstart;
	int linmaxiterend;

public:
	
	/// Sets required data and sets up the sparse Jacobian storage
	/** 
	 * \param[in] mesh Mesh context
	 * \param[in] spatial Spatial discretization context
	 * \param[in] soln Reference to the solution vector
	 * \param[in] pmat Jacobian matrix context for the preconditioner (and perhaps the solver)
	 * \param[in] cfl_init CFL to be used when the main solver starts
	 * \param[in] cfl_fin Final CFL number attained at \ref ramp_end iterations
	 * \param[in] ramp_start Iteration number of main solver at which 
	 *            CFL number begins to linearly increase
	 * \param[in] ramp_end Iteration number of the main solver at which CFL ramping ends
	 * \param[in] toler Relative residual tolerance for the ODE solver for the main solver
	 * \param[in] maxits Maximum number of pseudo-time steps for the main solver
	 * \param[in] linmaxiterstart Maximum iterations per time step for the linear solver
	 *              both for the starting ODE solver and initially for the main ODE solver
	 * \param[in] linmaxiterend Maximum iterations per time step at the end of the CFL ramping
	 *              of the main ODE solver
	 * \param[in] linearsolver Selects the linear solver to use; possible values:
	 *              "RICHARDSON", "BCGSTB" (BiCGStab)
	 * \param[in] precond Selects preconditioner to use for the linear solver; possible values:
	 *              "BSGS", "BILU0", "BJ"
	 * \param[in] restart_vecs Number of Krylov subspace vectors to store per restart iteration
	 * \param[in] log_nonlinear_res True if you want nonlinear convergence history
	 */
	SteadyBackwardEulerSolver(const UMesh2dh*const mesh, Spatial<nvars> *const spatial,
		MVector& soln, LinearOperator<a_real,a_int> *const pmat,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const double lin_tol, const int linmaxiterstart, 
		const int linmaxiterend, std::string linearsolver, std::string precond,
		const int restart_vecs, bool log_nonlinear_res);
	
	~SteadyBackwardEulerSolver();

	/// Runs the time-stepping loop
	/** Appends a line of timing-related data to a log file as follows.
	 *  num-cells num-threads  wall-time  CPU-time   avg-linear-solver-iterations 
	 *      number-of-time-steps  <\n>
	 * \param[in] logfile The file name to append timing data to
	 */
	void solve(std::string logfile);
};

}	// end namespace
#endif

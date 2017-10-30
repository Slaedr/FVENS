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

struct SteadySolverConfig {
	a_real cflinit;              ///< Initial CFL number, used for steps before \ref rampstart
	a_real cflfin;               ///< Final CFL, used for time steps after \ref rampend
	int rampstart;               ///< Time step at which to begin CFL ramping
	int rampend;                 ///< Time step at which to end CFL ramping
	a_real tol;                  ///< Tolerance for the final solution to the nonlinear system
	int maxiter;                 ///< Maximum number of iterations to solve the nonlinear system
	a_real lintol;               ///< Tolerance of the linear solver in case of implicit schemes
	int linmaxiterstart;         ///< Max linear solver iterations before step \ref rampstart
	int linmaxiterend;           ///< Max number of solver iterations after step \ref rampend
	bool lognres;                ///< Whether to output nonlinear residual history
	std::string logfile;         ///< File in which to write nonlinear residual history if needed
};

/// Base class for steady-state simulations in pseudo-time
/** Note that the unknowns u and residuals R correspond to the following ODE:
 * \f$ \frac{du}{dt} + R(u) = 0 \f$. Note that the residual is on the LHS.
 */
template <short nvars>
class SteadySolver
{
public:
	/** 
	 * \param[in] spatial Spatial discretization context
	 * \param[in] soln The solution vector to use and update
	 * \param[in] use_starter Whether to use \ref starterfv to generate an initial solution
	 * \param[in] log_nonlinear_residual Set to true if output of convergence history is needed
	 */
	SteadySolver(const Spatial<nvars> *const spatial, const SteadySolverConfig& conf)
		: space{spatial}, config{conf}, cputime{0.0}, walltime{0.0}, 
	{ }

	const MVector& residuals() const {
		return residual;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	virtual void solve(MVector& u) = 0;

	virtual ~SteadySolver() {}

protected:
	const Spatial<nvars> *const space;
	const SteadySolverConfig& config;
	MVector residual;
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
public:
	SteadyForwardEulerSolver(const Spatial<nvars> *const euler,
			const double toler, const int maxits, const double cfl,
			const bool use_implicitSmoothing, LinearOperator<a_real,a_int> *const A,
			bool log_nonlinear_res, const std::string log_file);
	
	~SteadyForwardEulerSolver();

	/// Solves the steady problem by a first-order explicit method, using local time-stepping
	/**
	 * \param[in,out] u The solution vector containing the initial solution and which
	 *   will contain the final solution on return.
	 */
	void solve(MVector& u);

private:
	using SteadySolver<nvars>::space;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;
	using SteadySolver<nvars>::lognres;
	using SteadySolver<nvars>::logfile;

	amat::Array2d<a_real> dtm;				///< Stores allowable local time step for each cell
	const double tol;
	const int maxiter;
	const double cfl;

	/// Stores whether implicit Laplacian smoothing is to be used for the residual
	const bool useImplicitSmoothing;

	/// Sparse matrix for the Laplacian
	LinearOperator<a_real,a_int> *const M;

	IterativeSolver<nvars> * linsolv;        ///< Linear solver context for Laplacian smoothing
	Preconditioner<nvars>* prec;             ///< preconditioner context for Laplacian smoothing
};

/// Implicit pseudo-time iteration to steady state
template <short nvars>
class SteadyBackwardEulerSolver : public SteadySolver<nvars>
{
	using SteadySolver<nvars>::space;
	using SteadySolver<nvars>::residual;
	using SteadySolver<nvars>::cputime;
	using SteadySolver<nvars>::walltime;
	using SteadySolver<nvars>::lognres;
	using SteadySolver<nvars>::logfile;

	amat::Array2d<a_real> dtm;               ///< Stores allowable local time step for each cell

	IterativeSolver<nvars> * linsolv;        ///< Linear solver context
	Preconditioner<nvars>* prec;             ///< preconditioner context

	/// Sparse matrix of the preconditioning Jacobian
	/** Note that the same matrix is used as the actual LHS as well,
	 * if matrix-free solution is disabled.
	 */
	LinearOperator<a_real,a_int>* M;


public:
	
	/// Sets required data and sets up the sparse Jacobian storage
	/** 
	 * \param[in] spatial Spatial discretization context
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
	SteadyBackwardEulerSolver(const Spatial<nvars> *const spatial,
		LinearOperator<a_real,a_int> *const pmat,
		const double cfl_init, const double cfl_fin, const int ramp_start, const int ramp_end, 
		const double toler, const int maxits, 
		const double lin_tol, const int linmaxiterstart, 
		const int linmaxiterend, std::string linearsolver, std::string precond,
		const int restart_vecs, bool log_nonlinear_res, const std::string log_file);
	
	~SteadyBackwardEulerSolver();

	/// Runs the time-stepping loop
	/** Appends a line of timing-related data to a log file as follows.
	 *  num-cells num-threads  wall-time  CPU-time   avg-linear-solver-iterations 
	 *      number-of-time-steps  <\n>
	 * \param[in,out] u The solution vector containing the initial solution and which
	 *   will contain the final solution on return.
	 */
	void solve(MVector& u);
};

/// Base class for unsteady simulations
/** Note that the unknowns u and residuals R correspond to the following ODE:
 * \f$ \frac{du}{dt} + R(u) = 0 \f$. Note that the residual is on the LHS.
 */
template <short nvars>
class UnsteadySolver
{
protected:
	const Spatial<nvars> * space;
	MVector& u;
	MVector residual;
	const int order;               ///< Deisgn order of accuracy in time
	double cputime;
	double walltime;
	const std::string logfile;

public:
	/** 
	 * \param[in] mesh Mesh context
	 * \param[in] spatial Spatial discretization context
	 * \param[in] soln The solution vector to use and update
	 * \param[in] temporal_order Design order of accuracy in time
	 */
	UnsteadySolver(const Spatial<nvars> *const spatial, MVector& soln,
			const int temporal_order, const std::string log_file)
		: space(spatial), u(soln), order{temporal_order}, cputime{0.0}, walltime{0.0},
		  logfile{log_file}
	{ }

	const MVector& residuals() const {
		return residual;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	/// Solve the ODE
	/**
	 */
	virtual void solve(const a_real time) = 0;

	virtual ~UnsteadySolver() {}
};

/// Total variation diminishing Runge-Kutta solvers upto order 3
template<short nvars>
class TVDRKSolver : public UnsteadySolver<nvars>
{
public:
	TVDRKSolver(const Spatial<nvars> *const spatial, MVector& soln,
			const int temporal_order, const std::string log_file, const double cfl_num);
	
	void solve(const a_real finaltime);

protected:
	using UnsteadySolver<nvars>::space;
	using UnsteadySolver<nvars>::residual;
	using UnsteadySolver<nvars>::u;
	using UnsteadySolver<nvars>::order;
	using UnsteadySolver<nvars>::cputime;
	using UnsteadySolver<nvars>::walltime;
	using UnsteadySolver<nvars>::logfile;

	const double cfl;

	/// Coefficients of TVD schemes
	const Matrix<a_real, Dynamic,Dynamic> tvdcoeffs;

private:
	amat::Array2d<a_real> dtm;
};
	

}	// end namespace
#endif

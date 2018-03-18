/** @file aodesolver.hpp
 * @brief Solution of ODEs resulting from some spatial discretization
 * @author Aditya Kashi
 * @date 24 Feb 2016; modified 13 May 2017
 */
#ifndef AODESOLVER_H
#define AODESOLVER_H 1

#include <vector>
#include <tuple>
#include <petscksp.h>
#include "aspatial.hpp"

namespace acfd {

/// A collection of parameters specifying the temporal discretization
struct SteadySolverConfig {
	bool lognres;                ///< Whether to output nonlinear residual history
	std::string logfile;         ///< File in which to write nonlinear residual history if needed
	a_real cflinit;              ///< Initial CFL number, used for steps before \ref rampstart
	a_real cflfin;               ///< Final CFL, used for time steps after \ref rampend
	int rampstart;               ///< Time step at which to begin CFL ramping
	int rampend;                 ///< Time step at which to end CFL ramping
	a_real tol;                  ///< Tolerance for the final solution to the nonlinear system
	int maxiter;                 ///< Maximum number of iterations to solve the nonlinear system
	int linmaxiterstart;         ///< Max linear solver iterations before step \ref rampstart
	int linmaxiterend;           ///< Max number of solver iterations after step \ref rampend
};

/// A collection of variables used for benchmarking purposes
struct TimingData {
	a_int nelem;                 ///< Size of the problem - the number of cells
	int num_threads;             ///< Number of threads used to solve the problem
	double lin_walltime;         ///< Wall-clock time taken by all the linear solves
	double lin_cputime;          ///< CPU time taken by all the linear solves
	double ode_walltime;         ///< Wall-clock time taken by the whole nonlinear ODE solve
	double ode_cputime;          ///< CPU time taken by the whole nonlinear ODE solve
	int total_lin_iters;         ///< Total number of linear iters needed for the ODE solve
	int avg_lin_iters;           ///< Average number of linear iters needed per time step
	int num_timesteps;           ///< Number of time steps needed for the ODE solve
};

/// Base class for steady-state simulations in pseudo-time
/** Note that the unknowns u and residuals r correspond to the following ODE:
 * \f$ \frac{du}{dt} - r(u) = 0 \f$.
 */
template <int nvars>
class SteadySolver
{
public:
	/** 
	 * \param[in] spatial Spatial discretization context
	 * \param[in] conf Reference to temporal discretization configuration settings
	 */
	SteadySolver(const Spatial<nvars> *const spatial, const SteadySolverConfig& conf);

	/// Get timing data
	TimingData getTimingData() const;

	/// Solve the nonlinear steady-state problem
	virtual StatusCode solve(Vec u) = 0;

	virtual ~SteadySolver() {}

protected:
	const Spatial<nvars> *const space;
	const SteadySolverConfig& config;
	Vec rvec;
	TimingData tdata;
};
	
/// A driver class for explicit time-stepping to steady state using forward Euler integration
/** \note Make sure compute_topological(), compute_face_data() and compute_areas()
 * have been called on the mesh object prior to initialzing an object of this class.
 * 
 * Optionally runs a `starter' time stepping loop to generate an initial solution
 * before starting the `main' loop.
 * The starter can perhaps use a first-order discretization.
 */
template <int nvars>
class SteadyForwardEulerSolver : public SteadySolver<nvars>
{
public:
	/// Sets the spatial context and problem configuration, and allocates required data
	/** \param x A PETSc Vec from which the residual vector is duplicated.
	 */
	SteadyForwardEulerSolver(const Spatial<nvars> *const euler, const Vec x, 
			const SteadySolverConfig& conf);
	
	~SteadyForwardEulerSolver();

	/// Solves the steady problem by a first-order explicit method, using local time-stepping
	/** Currently, the CFL number is constant and set to the 
	 * ['initial' CFL number](\ref SteadySolverConfig::cflinit).
	 * \param[in,out] u The solution vector containing the initial solution and which
	 *   will contain the final solution on return.
	 */
	StatusCode solve(Vec u);

private:
	using SteadySolver<nvars>::space;
	using SteadySolver<nvars>::config;
	using SteadySolver<nvars>::rvec;
	using SteadySolver<nvars>::tdata;

	std::vector<a_real> dtm;				///< Stores allowable local time step for each cell
};

/// Implicit pseudo-time iteration to steady state
template <int nvars>
class SteadyBackwardEulerSolver : public SteadySolver<nvars>
{
public:
	/// Sets required data and sets up the sparse Jacobian storage
	/** 
	 * \param[in] spatial Spatial discretization context
	 * \param[in] conf Temporal discretization settings
	 * \param[in] ksp The PETSc top-level solver context
	 */
	SteadyBackwardEulerSolver(const Spatial<nvars> *const spatial, const SteadySolverConfig& conf,
		KSP ksp);
	
	~SteadyBackwardEulerSolver();

	/// Runs the time-stepping loop with backward Euler time-stepping
	/** Stores timing data in a \ref TimingData object that can be retreived by \ref getTimingData. 
	 *
	 * Appends a line of timing-related data to a log file as follows.
	 *  num-cells num-threads  wall-time  CPU-time  total-lin-iterations  avg-linear-solver-iterations 
	 *      number-of-time-steps  <\n>
	 * \param[in,out] u The solution vector containing the initial solution and which
	 *   will contain the final solution on return.
	 */
	StatusCode solve(Vec u);

protected:
	using SteadySolver<nvars>::space;
	using SteadySolver<nvars>::config;
	using SteadySolver<nvars>::tdata;
	using SteadySolver<nvars>::rvec;       ///< Residual vector

	Vec duvec;                             ///< Nonlinear update vector
	std::vector<a_real> dtm;               ///< Stores allowable local time step for each cell

	KSP solver;                            ///< The solver context

	/// Linear CFL ramping 
	a_real linearRamp(const a_real cstart, const a_real cend, const int itstart, const int itend,
			const int itcur) const;

	/// A kind of exponential ramping, designed to be dependent on the residual ratio as base
	a_real expResidualRamp(const a_real cflmin, const a_real cflmax, const a_real prevcfl,
			const a_real resratio, const a_real paramup, const a_real paramdown);
};

/// Base class for unsteady simulations
/** Note that the unknowns u and residuals R correspond to the following ODE:
 * \f$ \frac{du}{dt} + R(u) = 0 \f$. Note that the residual is on the LHS.
 */
template <int nvars>
class UnsteadySolver
{
protected:
	const Spatial<nvars> * space;
	Vec uvec;
	Vec rvec;
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
	UnsteadySolver(const Spatial<nvars> *const spatial, Vec soln,
			const int temporal_order, const std::string log_file);

	/// Get timing data
	std::tuple<double,double> getRunTimes() const {
		return std::make_tuple(walltime, cputime);
	}

	/// Solve the ODE
	virtual StatusCode solve(const a_real time) = 0;

	virtual ~UnsteadySolver() {}
};

/// Total variation diminishing Runge-Kutta solvers upto order 3
template<int nvars>
class TVDRKSolver : public UnsteadySolver<nvars>
{
public:
	TVDRKSolver(const Spatial<nvars> *const spatial, Vec soln,
			const int temporal_order, const std::string log_file, const double cfl_num);

	~TVDRKSolver();
	
	StatusCode solve(const a_real finaltime);

protected:
	using UnsteadySolver<nvars>::space;
	using UnsteadySolver<nvars>::rvec;
	using UnsteadySolver<nvars>::uvec;
	using UnsteadySolver<nvars>::order;
	using UnsteadySolver<nvars>::cputime;
	using UnsteadySolver<nvars>::walltime;
	using UnsteadySolver<nvars>::logfile;

	const double cfl;

	/// Coefficients of TVD schemes
	const Matrix<a_real, Dynamic,Dynamic> tvdcoeffs;

private:
	std::vector<a_real> dtm;
};
	

}	// end namespace
#endif

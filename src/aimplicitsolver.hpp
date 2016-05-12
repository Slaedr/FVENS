/** @file aimplicitsolver.hpp
 * @brief Implicit solution of Euler/Navier-Stokes equations
 * @author Aditya Kashi
 * @date May 6, 2016
 */
#ifndef __AIMPLICITSOLVER_H

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX_H
#include <amatrix.hpp>
#endif

#ifndef __AMESH2DH_H
#include <amesh2dh.hpp>
#endif

#ifndef __ANUMERICALFLUX_H
#include <anumericalflux.hpp>
#endif

#ifndef __ALIMITER_H
#include <alimiter.hpp>
#endif

#ifndef __ARECONSTRUCTION_H
#include <areconstruction.hpp>
#endif

#ifndef __ALINALG_H
#include <alinalg.hpp>
#endif

#define __AIMPLICITSOLVER_H 1

namespace acfd {

/// Computation of the single-phase ideal gas Euler flux corresponding to any given state and along any given face-normal
class FluxFunction
{
protected:
	const acfd_real gamma;
public:
	FluxFunction (acfd_real _gamma) : gamma(_gamma)
	{ }

	void evaluate_flux(const amat::Matrix<acfd_real>& state, const acfd_real* const n, amat::Matrix<acfd_real>& flux) const
	{
		acfd_real vn = (state.get(1)*n[0] + state.get(2)*n[1])/state.get(0);
		acfd_real p = (gamma-1.0)*(state.get(3) - 0.5*(state.get(1)*state.get(1) + state.get(2)*state.get(2))/state.get(0));
		flux(0) = state.get(0) * vn;
		flux(1) = vn*state.get(1) + p*n[0];
		flux(2) = vn*state.get(2) + p*n[1];
		flux(3) = vn*(state.get(3) + p);
	}
};

/// A driver class to control the implicit time-stepping solution process
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of (a child class of) this class.
 */
class ImplicitSolver
{
protected:
	const UMesh2dh* const m;
	amat::Matrix<acfd_real> m_inverse;			///< mass matrix (just the volume of the element for FV)
	amat::Matrix<acfd_real> residual;			///< Right hand side for boundary integrals and source terms
	int nvars;									///< number of conserved variables
	amat::Matrix<acfd_real> uinf;				///< Free-stream/reference condition
	acfd_real g;								///< adiabatic index

	/// stores (for each cell i) \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face of the cell
	amat::Matrix<acfd_real> integ;

	/// Euler flux
	FluxFunction* eulerflux;

	/// Flux (boundary integral) calculation context
	InviscidFlux* inviflux;

	/// Reconstruction context
	Reconstruction* rec;

	/// Limiter context
	FaceDataComputation* lim;

	/// Cell centers
	amat::Matrix<acfd_real> rc;

	/// Ghost cell centers
	amat::Matrix<acfd_real> rcg;
	/// Ghost cell flow quantities
	amat::Matrix<acfd_real> ug;

	/// Number of Guass points per face
	int ngaussf;
	/// Faces' Gauss points' coords, stored a 3D array of dimensions naface x nguassf x ndim (in that order)
	amat::Matrix<acfd_real>* gr;

	/// Left state at each face (assuming 1 Gauss point per face)
	amat::Matrix<acfd_real> uleft;
	/// Rigt state at each face (assuming 1 Gauss point per face)
	amat::Matrix<acfd_real> uright;
	
	/// vector of unknowns
	amat::Matrix<acfd_real> u;
	/// x-slopes
	amat::Matrix<acfd_real> dudx;
	/// y-slopes
	amat::Matrix<acfd_real> dudy;

	/// Diagonal blocks of the residual Jacobian
	amat::Matrix<acfd_real>* diag;

	/// `Eigenvalues' of flux for LHS
	amat::Matix<acfd_real> lambdaij;

	/// Linear solver to use
	IterativeSolver* solver;

	amat::Matrix<acfd_real> scalars;		///< Holds density, Mach number and pressure for each cell
	amat::Matrix<acfd_real> velocities;		///< Holds velocity components for each cell

	int order;					///< Formal order of accuracy of the scheme (1 or 2)

	int solid_wall_id;			///< Boundary marker corresponding to solid wall
	int inflow_outflow_id;		///< Boundary marker corresponding to inflow/outflow
	const double cfl;
	const double w;

public:
	ImplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, const double cfl, const double relaxation_factor);
	~ImplicitSolver();
	void loaddata(acfd_real Minf, acfd_real vinf, acfd_real a, acfd_real rhoinf);

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Matrix<acfd_real>& instates, amat::Matrix<acfd_real>& bounstates);

	/// Calls functions to assemble the [right hand side](@ref residual)
	void compute_RHS();

	/// Compute diagonal blocks and eigenvalues of simplified flux for LHS
	virtual void compute_LHS();
	
	/// Computes the left and right states at each face, using the [reconstruction](@ref rec) and [limiter](@ref limiter) objects
	void compute_face_states();

	virtual void solve() = 0;

	/// Computes the L2 norm of a cell-centered quantity
	acfd_real l2norm(const amat::Matrix<acfd_real>* const v);
	
	/// Compute cell-centred quantities to export
	void postprocess_cell();
	
	/// Compute nodal quantities to export, based on area-weighted averaging (which takes into account ghost cells as well)
	void postprocess_point();

	/// Compute norm of cell-centered entropy production
	/// Call aftr computing pressure etc \sa postprocess_cell
	acfd_real compute_entropy_cell();

	amat::Matrix<acfd_real> getscalars() const;
	amat::Matrix<acfd_real> getvelocities() const;

	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint();

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face();
};

/// Solves for a steady-state solution by a first-order implicit scheme
/** The ODE system is linearized at each time step; equivalent to a single Newton iteration at each time step.
 */
class SteadyStateImplicitSolver : public ImplicitSolver
{
	const acfd_real lintol;
	const acfd_real steadytol;
	const int linmaxiter;
	const int steadymaxiter;
public:
	SteadyStateImplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, const double cfl, const double omega,
			const acfd_real lin_tol, const int lin_maxiter, const acfd_real steady_tol, const int steady_maxiter);

	/// Solves a steady problem by an implicit method first order in time, using local time-stepping
	void solve();
};

class UnsteadyImplicitSolver : public ImplicitSolver
{
	const acfd_real lintol;
	const acfd_real newtontol;
	const int linmaxiter;
	const int newtonmaxiter;
public:
	UnsteadyImplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, const double cfl, const double omega,
			const acfd_real lin_tol, const int lin_maxiter, const acfd_real newton_tol, const int newton_maxiter);

	/// Solves unsteady problem
	virtual void solve() = 0;
};

/// Solves an unsteady problem using first-order implicit time-stepping
class BackwardEulerSolver : public UnsteadyImplicitSolver
{
public:
	BackwardEulerSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter, std::string linear_solver, const double cfl, const double omega,
			const acfd_real lin_tol, const int lin_maxiter, const acfd_real newton_tol, const int newton_maxiter);

	void solve();
};

}	// end namespace
#endif

/** @file aexplicitsolver.hpp
 * @brief Implements a driver class for explicit solution of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 */
#ifndef __AEXPLICITSOLVER_H

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

#define __AEXPLICITSOLVER_H 1

namespace acfd {

/// A driver class to control the explicit time-stepping solution using TVD Runge-Kutta time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class ExplicitSolver
{
	const UMesh2dh* m;
	amat::Matrix<acfd_real> m_inverse;			///< Left hand side (just the volume of the element for FV)
	amat::Matrix<acfd_real> residual;			///< Right hand side for boundary integrals and source terms
	int nvars;									///< number of conserved variables ** deprecated, use the preprocessor constant NVARS instead **
	amat::Matrix<acfd_real> uinf;				///< Free-stream/reference condition
	acfd_real g;								///< adiabatic index

	/// stores (for each cell i) \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face of the cell
	amat::Matrix<acfd_real> integ;

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

	/// Flux across each face
	amat::Matrix<acfd_real> fluxes;
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

	amat::Matrix<acfd_real> scalars;		///< Holds density, Mach number and pressure for each cell
	amat::Matrix<acfd_real> velocities;		///< Holds velocity components for each cell

	int order;					///< Formal order of accuracy of the scheme (1 or 2)

	int solid_wall_id;			///< Boundary marker corresponding to solid wall
	int inflow_outflow_id;		///< Boundary marker corresponding to inflow/outflow

public:
	ExplicitSolver(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter);
	~ExplicitSolver();
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
	
	/// Computes the left and right states at each face, using the [reconstruction](@ref rec) and [limiter](@ref limiter) objects
	void compute_face_states();

	/// Solves a steady problem by an explicit method first order in time, using local time-stepping
	void solve_rk1_steady(const acfd_real tol, const int maxiter, const acfd_real cfl);

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

}	// end namespace
#endif

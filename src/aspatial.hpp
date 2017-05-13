/** @file aspatial.hpp
 * @brief Spatial discretization for Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016; modified May 13 2017
 */
#ifndef __ASPATIAL_H
#define __ASPATIAL_H 1

#ifndef __ACONSTANTS_H
#include "aconstants.hpp"
#endif

#ifndef __AMATRIX_H
#include "amatrix.hpp"
#endif

#ifndef __AMESH2DH_H
#include "amesh2dh.hpp"
#endif

#ifndef __ANUMERICALFLUX_H
#include "anumericalflux.hpp"
#endif

#ifndef __ALIMITER_H
#include "alimiter.hpp"
#endif

#ifndef __ARECONSTRUCTION_H
#include "areconstruction.hpp"
#endif

namespace acfd {

/// A driver class to control the explicit time-stepping solution using TVD Runge-Kutta time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class EulerFV
{
protected:
	const UMesh2dh* m;
	amat::Matrix<a_real> m_inverse;			///< Left hand side (just the volume of the element for FV)
	amat::Matrix<a_real> residual;			///< Right hand side for boundary integrals and source terms
	int nvars;									///< number of conserved variables ** deprecated, use the preprocessor constant NVARS instead **
	amat::Matrix<a_real> uinf;				///< Free-stream/reference condition
	a_real g;								///< adiabatic index

	/// stores (for each cell i) \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face of the cell
	amat::Matrix<a_real> integ;
	
	/// Stores allowable local time step for each cell
	amat::Matrix<a_real> dtm;

	/// Analytical flux vector computation
	EulerFlux aflux;
	
	/// Numerical inviscid flux calculation context
	InviscidFlux* inviflux;

	/// Reconstruction context
	Reconstruction* rec;

	/// Limiter context
	FaceDataComputation* lim;

	/// Cell centers
	amat::Matrix<a_real> rc;

	/// Ghost cell centers
	amat::Matrix<a_real> rcg;
	/// Ghost cell flow quantities
	amat::Matrix<a_real> ug;

	/// Number of Guass points per face
	int ngaussf;
	/// Faces' Gauss points' coords, stored a 3D array of dimensions naface x nguassf x ndim (in that order)
	amat::Matrix<a_real>* gr;

	/// Flux across each face
	amat::Matrix<a_real> fluxes;
	/// Left state at each face (assuming 1 Gauss point per face)
	amat::Matrix<a_real> uleft;
	/// Rigt state at each face (assuming 1 Gauss point per face)
	amat::Matrix<a_real> uright;
	
	/// vector of unknowns
	amat::Matrix<a_real> u;
	/// x-slopes
	amat::Matrix<a_real> dudx;
	/// y-slopes
	amat::Matrix<a_real> dudy;

	int order;								///< Formal order of accuracy of the scheme (1 or 2)

	int solid_wall_id;						///< Boundary marker corresponding to solid wall
	int inflow_outflow_id;					///< Boundary marker corresponding to inflow/outflow
	
	amat::Matrix<a_real> scalars;			///< Holds density, Mach number and pressure for each cell
	amat::Matrix<a_real> velocities;		///< Holds velocity components for each cell

public:
	EulerFV(const UMesh2dh* mesh, const int _order, std::string invflux, std::string reconst, std::string limiter);
	~EulerFV();
	void loaddata(a_real Minf, a_real vinf, a_real a, a_real rhoinf);

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Matrix<a_real>& instates, amat::Matrix<a_real>& bounstates);

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** This invokes flux calculating function after zeroing the residuals
	 * and also computes [local time steps](@ref dtm).
	 */
	void compute_residual();
	
	/// Computes the left and right states at each face, using the [reconstruction](@ref rec) and [limiter](@ref limiter) objects
	void compute_face_states();

	/// Computes the L2 norm of a cell-centered quantity
	a_real l2norm(const amat::Matrix<a_real>* const v);
	
	/// Compute cell-centred quantities to export
	void postprocess_cell();
	
	/// Compute nodal quantities to export, based on area-weighted averaging (which takes into account ghost cells as well)
	void postprocess_point();

	/// Compute norm of cell-centered entropy production
	/// Call aftr computing pressure etc \sa postprocess_cell
	a_real compute_entropy_cell();

	amat::Matrix<a_real> getscalars() const;
	amat::Matrix<a_real> getvelocities() const;

	const amat::Matrix<a_real>& localTimeSteps() const {
		return dtm;
	}
	
	const amat::Matrix<a_real>& residuals() const {
		return residual;
	}
	
	/// Write access to the conserved variables
	amat::Matrix<a_real>& unknowns() {
		return u;
	}

	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint();

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face();
};

}	// end namespace
#endif

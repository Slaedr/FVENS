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

#ifndef __AARRAY2D_H
#include "aarray2d.hpp"
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

#if HAVE_PETSC==1
#include <petscmat.h>
#endif

namespace acfd {

/// Base class for finite volume spatial discretization
template<short nvars>
class Spatial
{
protected:
	/// Mesh context
	const UMesh2dh *const m;

	/// Cell centers
	amat::Array2d<a_real> rc;

	/// Ghost cell centers
	amat::Array2d<a_real> rcg;

	/// Faces' Gauss points' coords, stored a 3D array of dimensions 
	/// naface x nguass x ndim (in that order)
	amat::Array2d<a_real>* gr;
	
	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint();

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face();

	/// step length for finite difference Jacobian
	const a_real eps;

public:
	/// Common setup required for finite volume discretizations
	/** Computes and stores cell centre coordinates, ghost cells' centres, and 
	 * quadrature point coordinates.
	 */
	Spatial(const UMesh2dh *const mesh);

	virtual ~Spatial();
	
	/// Computes the residual and local time steps
	/** \param[in] u The state at which the residual is to be computed
	 * \param[in|out] residual The residual is added to this
	 * \param[in] gettimesteps Whether time-step computation is required
	 * \param[out] dtm Local time steps are stored in this
	 */
	virtual void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) = 0;
	
	/// Computes the Jacobian matrix of the residual
	virtual void compute_jacobian(const MVector& u, LinearOperator<a_real,a_int> *const A) = 0;

	/// Computes the Frechet derivative of the residual along a given direction 
	/// using finite difference
	/** \param[in] resu The residual vector at the state at which the derivative is to be computed
	 * \param[in] u The state at which the derivative is to be computed
	 * \param[in] v The direction in which the derivative is to be computed
	 * \param[in] add_time_deriv Whether or not time-derivative term is to be added
	 * \param[in] dtm Vector of local time steps for time derivative term
	 * \param aux Storage for intermediate state
	 * \param[out] prod The vector containing the directional derivative
	 */
	virtual void compute_jac_vec(const MVector& resu, const MVector& u, 
			const MVector& v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			MVector& __restrict aux,
			MVector& __restrict prod);
	
	/// Computes a([M du/dt +] dR/du) v + b w and stores in prod
	virtual void compute_jac_gemv(const a_real a, const MVector& resu, const MVector& u, 
			const MVector& v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const MVector& w,
			MVector& __restrict aux,
			MVector& __restrict prod);
};

/// Computes the integrated fluxes and their Jacobians for compressible flow
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() 
 * have been called on the mesh object prior to initialzing an object of this class.
 */
class FlowFV : public Spatial<NVARS>
{
public:
	/// Sets data and various numerics objects
	/** \param[in] mesh The mesh context
	 * \param[in] g Adiabatic index
	 * \param[in] Minf Free-stream Mach number
	 * \param[in] Tinf Free stream dimensional temperature
	 * \param[in] Reinf Free-stream Reynolds number
	 * \param[in] Pr Prandtl number
	 * \param[in] isothermal_marker The boundary marker in the mesh file corresponding to
	 *   isothermal wall boundaries
	 * \param[in] isothermal_wall_temperature Wall temperature boundary value in Kelvin; this is
	 *   divided by free-stream temperature in this routine and the non-dimensional value is stored
	 * \param[in] invflux The inviscid flux to use - VANLEER, HLL, HLLC
	 * \param[in] jacflux The inviscid flux to use for computing the first-order Jacobian
	 * \param[in] reconst The method used for gradient reconstruction 
	 *   - NONE, GREENGAUSS, LEASTSQUARES
	 * \param[in] limiter The kind of slope limiter to use
	 */
	FlowFV(const UMesh2dh *const mesh, const a_real g, const a_real Minf, const a_real Tinf, 
		const a_real Reinf, const a_real Pr,
		const int isothermal_marker, const int adiabatic_marker, const int slip_marker,
		const int inflowoutflow_marker, const a_real isothermal_wall_temperature,
		std::string invflux, std::string jacflux, std::string reconst, std::string limiter);
	
	~FlowFV();

	/// Sets initial conditions
	/** \param[in] a Angle of attack in radians
	 */
	void loaddata(const a_real a, MVector& u);

	/// Set simulation data for special cases
	void loaddata_special(const short inittype, const a_real vinf, const a_real a, 
			const a_real rhoinf, MVector& u);

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** This invokes flux calculation after zeroing the residuals and also computes local time steps.
	 */
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm);

#if HAVE_PETSC==1
	/// Computes the residual Jacobian as a PETSc martrix
	void compute_jacobian(const MVector& u, const bool blocked, Mat A);
#else
	/// Computes the residual Jacobian as arrays of diagonal blocks for each cell, 
	/// and lower and upper blocks for each face
	/** A is not zeroed before use.
	 */
	void compute_jacobian(const MVector& u, LinearOperator<a_real,a_int> *const A);
#endif

	/// Compute cell-centred quantities to export
	void postprocess_cell(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& velocities);
	
	/// Compute nodal quantities to export
	/** Based on area-weighted averaging which takes into account ghost cells as well
	 */
	void postprocess_point(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& velocities);

	/// Compute norm of cell-centered entropy production
	/** Call aftr computing pressure etc \sa postprocess_cell
	 */
	a_real compute_entropy_cell(const MVector& u);

protected:
	amat::Array2d<a_real> uinf;				///< Free-stream/reference condition

	/// Analytical flux vector computation
	IdealGasPhysics physics;
	
	/// Numerical inviscid flux calculation context for residual computation
	/** This is the "actual" flux being used.
	 */
	InviscidFlux* inviflux;

	/// Numerical inviscid flux context for the Jacobian
	InviscidFlux* jflux;

	/// Reconstruction context
	Reconstruction* rec;
	
	bool secondOrderRequested;

	bool allocflux;

	bool reconstructPrimitive;

	/// Limiter context
	FaceDataComputation* lim;

	const int isothermal_wall_id;				///< Boundary marker for isothermal wall
	const int adiabatic_wall_id;				///< Boundary marker for adiabatic wall
	const int slip_wall_id;						///< Boundary marker corresponding to solid wall
	const int inflow_outflow_id;				///< Boundary marker corresponding to inflow/outflow
	
	/// Inflow boundary marker for supersonic vortex case
	const int supersonic_vortex_case_inflow;

	const a_real isothermal_wall_temp;		///< Temperature imposed at isothermal wall

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) 
	/// using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates);

	/// Computes ghost cell state across the face denoted by the first parameter
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs);
};

/// Spatial discretization of diffusion operator with constant difusivity
template <short nvars>
class Diffusion : public Spatial<nvars>
{
protected:
	using Spatial<nvars>::m;
	using Spatial<nvars>::rc;
	using Spatial<nvars>::rcg;
	using Spatial<nvars>::gr;
	const a_real diffusivity;		///< Diffusion coefficient (eg. kinematic viscosity)
	const a_real bval;				///< Dirichlet boundary value
	
	/// Pointer to a function that describes the  source term
	std::function <
		void(const a_real *const, const a_real, const a_real *const, a_real *const)
					> source;

	std::vector<a_real> h;			///< Size of cells
	
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs);
	
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates);

public:
	Diffusion(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
			> source);
	
	virtual void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) = 0;
	
	virtual void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A) = 0;

	virtual void postprocess_point(const MVector& u, amat::Array2d<a_real>& up);

	virtual ~Diffusion();
};

/// Spatial discretization of diffusion operator with constant diffusivity 
/// using `modified gradient' or `corrected gradient' method
template <short nvars>
class DiffusionMA : public Diffusion<nvars>
{
	using Spatial<nvars>::m;
	using Spatial<nvars>::rc;
	using Spatial<nvars>::rcg;
	using Spatial<nvars>::gr;
	using Diffusion<nvars>::diffusivity;
	using Diffusion<nvars>::bval;
	using Diffusion<nvars>::source;
	using Diffusion<nvars>::h;
	
	using Diffusion<nvars>::compute_boundary_state;
	using Diffusion<nvars>::compute_boundary_states;
	
	Reconstruction* rec;
	amat::Array2d<a_real> dudx;				///< X-gradients at cell centres
	amat::Array2d<a_real> dudy;				///< Y-gradients at cell centres
	amat::Array2d<a_real> uleft;			///< Left state at each face
	amat::Array2d<a_real> uright;			///< Right state at each face
	amat::Array2d<a_real> ug;				///< Boundary states

public:
	DiffusionMA(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
				> source, 
			std::string reconst);
	
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm);
	
	void add_source(const MVector& u, 
			MVector& __restrict residual, amat::Array2d<a_real>& __restrict dtm);
	
	void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A);

	~DiffusionMA();
	
	using Diffusion<nvars>::postprocess_point;
};

}	// end namespace
#endif

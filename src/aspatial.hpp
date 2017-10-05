/** @file aspatial.hpp
 * @brief Spatial discretization for Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016; modified May 13 2017
 */
#ifndef ASPATIAL_H
#define ASPATIAL_H 1

#include "aconstants.hpp"

#include "aarray2d.hpp"

#include "amesh2dh.hpp"

#include "anumericalflux.hpp"
#include "alimiter.hpp"
#include "areconstruction.hpp"

#if HAVE_PETSC==1
#include <petscmat.h>
#endif

namespace acfd {

/// Base class for finite volume spatial discretization
template<short nvars>
class Spatial
{
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
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const = 0;
	
	/// Computes the Jacobian matrix of the residual
	virtual void compute_jacobian(const MVector& u, LinearOperator<a_real,a_int> *const A) const = 0;

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

	/// Computes gradients of field variables and stores them in the argument
	virtual void getGradients(const MVector& u, MVector grad[NDIM]) const = 0;

protected:
	/// Mesh context
	const UMesh2dh *const m;

	/// Cell centers of both real cells and ghost cells
	amat::Array2d<a_real> rc;

	/// Faces' Gauss points' coords, stored a 3D array of dimensions 
	/// naface x nguass x ndim (in that order)
	amat::Array2d<a_real>* gr;
	
	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint(amat::Array2d<a_real>& rchg);

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face(amat::Array2d<a_real>& rchg);

	/// step length for finite difference Jacobian
	const a_real eps;
};

/// Computes the integrated fluxes and their Jacobians for compressible flow
/** Note about BCs: normal velocity is assumed zero at all walls.
 * \note Make sure compute_topological(), compute_face_data() and compute_jacobians() 
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
	 * \param[in] aoa Angle of attack in radians
	 * \param[in] compute_viscous Set to true if viscous fluxes are required
	 * \param[in] useConstVisc Set true to use constant free-stream viscosity throughout
	 * \param[in] isothermal_marker The boundary marker in the mesh file corresponding to
	 *   isothermal wall boundaries
	 * \param[in] farfield_marker ID for boundaries where farfield conditions are imposed in ghost
	 *   cells
	 * \param[in] inflowoutflow_marker ID for boundaries where farfield condititions are imposed in
	 *   ghost cells of inflow boundaries, and only farfield pressure is imposed in ghost cells
	 *   of outflow boundaries
	 * \param[in] extrap_marker Boundary marker for extrapolation BC
	 * \param[in] isothermal_Temperature Wall temperature boundary value in Kelvin; this is
	 *   divided by free-stream temperature in this routine and the non-dimensional value is stored
	 * \param[in] isothermal_TangVel Tangential non-dimensional velocity at isothermal boundaries
	 * \param[in] invflux The inviscid flux to use - VANLEER, HLL, HLLC
	 * \param[in] jacflux The inviscid flux to use for computing the first-order Jacobian
	 * \param[in] reconst The method used for gradient reconstruction 
	 *   - NONE, GREENGAUSS, LEASTSQUARES
	 * \param[in] limiter The kind of slope limiter to use
	 * \param[in] order2 True if second-order solution is desired.
	 */
	FlowFV(const UMesh2dh *const mesh, const a_real g, const a_real Minf, const a_real Tinf, 
		const a_real Reinf, const a_real Pr, const a_real aoa, 
		const bool compute_viscous, const bool useConstVisc,
		const int isothermal_marker, const int adiabatic_marker, const int isothermalbaric_marker, 
		const int slip_marker, const int farfield_marker, const int inflowoutflow_marker,
		const int extrap_marker,
		const a_real isothermal_Temperature, const a_real isothermal_TangVel, 
		const a_real adiabisobaric_Temperature, const a_real adiabisobaric_TangVel, 
		const a_real adiabisobaric_Pressure,
		const a_real adiabatic_TangVel,
		std::string invflux, std::string jacflux, std::string reconst, std::string limiter,
		const bool order2, const bool reconst_prim);
	
	~FlowFV();

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	void initializeUnknowns(const bool fromfile, const std::string file, MVector& u);

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** This invokes flux calculation after zeroing the residuals and also computes local time steps.
	 */
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const;

#if HAVE_PETSC==1
	/// Computes the residual Jacobian as a PETSc martrix
	void compute_jacobian(const MVector& u, const bool blocked, Mat A);
#else
	/// Computes the residual Jacobian as arrays of diagonal blocks for each cell, 
	/// and lower and upper blocks for each face
	/** A is not zeroed before use.
	 */
	void compute_jacobian(const MVector& u, LinearOperator<a_real,a_int> *const A) const;
#endif
	
	void getGradients(const MVector& u, MVector grad[NDIM]) const;

	/// Compute cell-centred quantities to export
	void postprocess_cell(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& velocities);
	
	/// Compute nodal quantities to export
	/** Based on area-weighted averaging which takes into account ghost cells as well.
	 * Density, Mach number, pressure and temperature are the exported scalars,
	 * and velocity is exported as well.
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

	/// If true, compute Navier Stokes fluxes, else only Euler
	const bool computeViscous;
	
	/// If true, use constant viscosity rather than that given by gas physics context
	const bool constVisc;

	/// Numerical inviscid flux calculation context for residual computation
	/** This is the "actual" flux being used.
	 */
	InviscidFlux* inviflux;

	/// Numerical inviscid flux context for the Jacobian
	InviscidFlux* jflux;

	/// Reconstruction context
	Reconstruction* rec;

	bool allocflux;

	/// Limiter context
	FaceDataComputation* lim;

	const int isothermal_wall_id;				///< Boundary marker for isothermal wall
	const int adiabatic_wall_id;				///< Boundary marker for adiabatic wall
	const int isothermalbaric_wall_id;			/**< Marker for adiabatic wall with a pressure value 
													additionally imposed */
	const int slip_wall_id;						///< Boundary marker corresponding to solid wall
	const int farfield_id;				///< Boundary marker corresponding to farfield
	const int inflowoutflow_id;			///< Boundary marker corresponding to inflow/outflow
	const int extrap_id;						///< Marker for extrapolation boundary

	const a_real isothermal_wall_temperature;      ///< Temperature imposed at isothermal wall
	const a_real isothermal_wall_tangvel;          ///< Tangential velocity at isothermal wall
	const a_real adiabatic_wall_tangvel;           ///< Tangential velocity at adiabatic wall
	const a_real isothermalbaric_wall_temperature;  ///< Temperature at isothermal isobaric wall
	const a_real isothermalbaric_wall_tangvel;  ///< Tangential velocity at isothermal isobaric wall
	const a_real isothermalbaric_wall_pressure; ///< Pressure imposed at isothermal isobaric wall
	
	const bool secondOrderRequested;					///< True if reconstruction is to be used

	/// True if primitive variables should be reconstructed rather than conserved variables
	const bool reconstructPrimitive;

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) 
	/// using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates) const;

	/// Computes ghost cell state across a face
	/** \param[in] ied Face id in face data structure intfac
	 * \param[in] ins Interior state of conserved variables
	 * \param[in|out] Ghost state of conserved variables
	 */
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs) const;

	/// Computes viscous flux across a face
	/** The output vflux needs to be integrated on the face.
	 * \param[in] iface Face index
	 * \param[in] u Cell-centred conserved variables
	 * \param[in] ug Ghost cell-centred conserved variables
	 * \param[in] dudx Cell-centred gradients ("optional")
	 * \param[in] dudy Cell-centred gradients ("optional", see below)
	 * \param[in] ul Left state of faces (conserved variables)
	 * \param[in] ul Right state of faces (conserved variables)
	 * \param[in|out] vflux On output, contains the viscous flux across the face
	 *
	 * Note that dudx and dudy can be unallocated if only first-order fluxes are being computed,
	 * but ul and ur are always used.
	 */
	void computeViscousFlux(const a_int iface, const MVector& u, const amat::Array2d<a_real>& ug,
			const amat::Array2d<a_real>& dudx, const amat::Array2d<a_real>& dudy,
			const amat::Array2d<a_real>& ul, const amat::Array2d<a_real>& ur,
			a_real *const vflux) const;

	/// Compues the first-order "thin-layer" viscous flux Jacobian
	/** This is the same sign as is needed in the residual; note that the viscous flux Jacobian is
	 * added to the output matrices - they are not zeroed or directly assigned to.
	 * The outputs vfluxi and vfluxj still need to be integrated on the face.
	 * \param[in] iface Face index
	 * \param[in] ul Cell-centred conserved variable on left
	 * \param[in] ur Cell-centred conserved variable on right
	 * \param[in|out] vfluxi Flux Jacobian \f$ \partial \mathbf{f}_{ij} / \partial \mathbf{u}_i \f$
	 *   NVARS x NVARS array stored as a 1D row-major array
	 * \param[in|out] vfluxj Flux Jacobian \f$ \partial \mathbf{f}_{ij} / \partial \mathbf{u}_j \f$
	 *   NVARS x NVARS array stored as a 1D row-major array
	 */
	void computeViscousFluxJacobian(const a_int iface,
			const a_real *const ul, const a_real *const ur,
			a_real *const __restrict vfluxi, a_real *const __restrict vfluxj) const;
};

/// Spatial discretization of diffusion operator with constant difusivity
template <short nvars>
class Diffusion : public Spatial<nvars>
{
public:
	Diffusion(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
			> source);

	/// Sets initial conditions to zero
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	void initializeUnknowns(const bool fromfile, const std::string file, MVector& u) const;
	
	virtual void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const = 0;
	
	virtual void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A) const = 0;
	
	virtual void getGradients(const MVector& u, MVector grad[NDIM]) const = 0;

	virtual void postprocess_point(const MVector& u, amat::Array2d<a_real>& up);

	virtual ~Diffusion();

protected:
	using Spatial<nvars>::m;
	using Spatial<nvars>::rc;
	using Spatial<nvars>::gr;
	const a_real diffusivity;		///< Diffusion coefficient (eg. kinematic viscosity)
	const a_real bval;				///< Dirichlet boundary value
	
	/// Pointer to a function that describes the  source term
	std::function <
		void(const a_real *const, const a_real, const a_real *const, a_real *const)
					> source;

	std::vector<a_real> h;			///< Size of cells
	
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs) const;
	
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates) const;
};

/// Spatial discretization of diffusion operator with constant diffusivity 
/// using `modified gradient' or `corrected gradient' method
template <short nvars>
class DiffusionMA : public Diffusion<nvars>
{
public:
	DiffusionMA(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
				> source, 
			std::string reconst);
	
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const;
	
	/*void add_source(const MVector& u, 
			MVector& __restrict residual, amat::Array2d<a_real>& __restrict dtm) const;*/
	
	void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A) const;
	
	void getGradients(const MVector& u, MVector grad[NDIM]) const;
	
	using Diffusion<nvars>::initializeUnknowns;

	~DiffusionMA();

protected:
	using Diffusion<nvars>::postprocess_point;
	using Spatial<nvars>::m;
	using Spatial<nvars>::rc;
	using Spatial<nvars>::gr;
	using Diffusion<nvars>::diffusivity;
	using Diffusion<nvars>::bval;
	using Diffusion<nvars>::source;
	using Diffusion<nvars>::h;
	
	using Diffusion<nvars>::compute_boundary_state;
	using Diffusion<nvars>::compute_boundary_states;
	
	Reconstruction* rec;
};

}	// end namespace
#endif

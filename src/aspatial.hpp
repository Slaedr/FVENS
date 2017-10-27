/** @file aspatial.hpp
 * @brief Spatial discretization for Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016; modified May 13 2017
 */
#ifndef ASPATIAL_H
#define ASPATIAL_H 1

#include <array>

#include "aconstants.hpp"

#include "aarray2d.hpp"

#include "amesh2dh.hpp"
#include "anumericalflux.hpp"
#include "agradientschemes.hpp"
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

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	virtual void initializeUnknowns(MVector& u) const = 0;
	
	/// Compute nodal quantities to export
	virtual void postprocess_point(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& vector) const = 0;

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

/// The collection of physical data needed to initialize flow spatial discretizations
struct FlowPhysicsConfig
{
	const a_real gamma;                       ///< Adiabatic index 
	const a_real Minf;                        ///< Free-stream Mach number
	const a_real Tinf;                        ///< Free-stream temperature in Kelvin
	const a_real Reinf;                       ///< Free-stream Reynolds number
	const a_real Pr;                          ///< (Constant) Prandtl number
	const a_real aoa;                         ///< Angle of attack in radians 
	const bool viscous_sim;                   ///< Whether to include viscous effects
	const bool const_visc;                  ///< Whether to use constant viscosity
	const int isothermalwall_id;            ///< Boundary marker for isothermal no-slip wall
	const int adiabaticwall_id;             ///< Marker for adiabatic no-slip wall
	const int isothermalbaricwall_id;       ///< Marker for isothermal fixed-pressure no-slip wall
	const int slipwall_id;                  ///< Marker for slip wall
	const int farfield_id;                  ///< Marker for far-field boundary
	const int inflowoutflow_id;             ///< Marker for inflow/outflow boundary
	const int extrapolation_id;             ///< Marker for an extrapolation boundary
	const int periodic_id;                  ///< Marker for periodic boundary
	const a_real isothermalwall_temp;       ///< Temperature at isothermal wall
	const a_real isothermalwall_vel;        ///< Tangential velocity at isothermal wall 
	const a_real adiabaticwall_vel;         ///< Tangential velocity at adiabatic wall
	const a_real isothermalbaricwall_temp;  ///< Temperature at isothermal fixed-pressure wall
	const a_real isothermalbaricwall_vel;   ///< Tangential velocity at isothermal pressure wall
};

/// Collection of options related to the spatial discretization scheme
struct FlowNumericsConfig
{
	const std::string conv_numflux;         ///< Convective numerical flux to use
	const std::string conv_numflux_jac;     ///< Conv. numer. flux to use for approximate Jacobian
	const std::string gradientscheme;       ///< Method to use to compute gradients
	const std::string reconstruction;       ///< Method to use to reconstruct the solution
	const bool order2;                      ///< Whether to compute a second-order solution
};

/// Computes the integrated fluxes and their Jacobians for compressible flow
/** Note about BCs: normal velocity is assumed zero at all walls.
 * \note Make sure compute_topological(), compute_face_data() and compute_jacobians() 
 * have been called on the mesh object prior to initialzing an object of this class.
 */
template <
	bool secondOrderRequested,        ///< Whether to computes gradients to get a 2nd order solution
	bool constVisc                    ///< Whether to use constant viscosity (true) or Sutherland
>
class FlowFV : public Spatial<NVARS>
{
public:
	/// Sets data and initializes the numerics
	/** \note The parameter for the Venkatakrishnan limiter is set here, for some reason.
	 */
	FlowFV(const UMesh2dh *const mesh,                  ///< Mesh context
		const FlowPhysicsConfig& pconfiguration,        ///< Physical data defining the problem
		const FlowNumericsConfig& nconfiguration);      ///< Options defining the numerical method
	
	~FlowFV();

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	void initializeUnknowns(MVector& u) const;

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** This invokes flux calculation after zeroing the residuals and also computes local time steps.
	 */
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const;

#if HAVE_PETSC==1
	/// Computes the residual Jacobian as a PETSc martrix
	void compute_jacobian(const MVector& u, const bool blocked, Mat A) const;
#else
	/// Computes the residual Jacobian as arrays of diagonal blocks for each cell, 
	/// and lower and upper blocks for each face
	/** Periodic boundary conditions are not linearized, and as such,
	 * implicit solution of problems with periodic boundaries should not be attempted.
	 * Also, A is not zeroed before use.
	 */
	void compute_jacobian(const MVector& u, LinearOperator<a_real,a_int> *const A) const;
#endif
	
	void getGradients(const MVector& u, MVector grad[NDIM]) const;

	/// Compute cell-centred quantities to export
	void postprocess_cell(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& velocities) const;
	
	/// Compute nodal quantities to export
	/** Based on area-weighted averaging which takes into account ghost cells as well.
	 * Density, Mach number, pressure and temperature are the exported scalars,
	 * and velocity is exported as well.
	 */
	void postprocess_point(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& velocities) const;

	/// Compute norm of cell-centered entropy production
	/** Call aftr computing pressure etc \sa postprocess_cell
	 */
	a_real compute_entropy_cell(const MVector& u) const;

protected:
	/// Problem specification
	const FlowPhysicsConfig& pconfig;

	/// Numerical method specification
	const FlowNumericsConfig& nconfig;

	/// Analytical flux vector computation
	const IdealGasPhysics physics;
	
	const std::array<a_real,NVARS> uinf;                    ///< Free-stream/reference condition

	/// Numerical inviscid flux calculation context for residual computation
	/** This is the "actual" flux being used.
	 */
	const InviscidFlux *const inviflux;

	/// Numerical inviscid flux context for the Jacobian
	const InviscidFlux *const jflux;

	/// Gradient computation context
	const GradientScheme *const gradcomp;

	/// Reconstruction context
	const SolutionReconstruction *const lim;

	/// Computes flow variables at all boundaries (either Gauss points or ghost cell centers) 
	/// using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates) const;

	/// Computes ghost cell state across one face
	/** \param[in] ied Face id in face data structure intfac
	 * \param[in] ins Interior state of conserved variables
	 * \param[in|out] gs Ghost state of conserved variables
	 */
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const gs) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	/** The output array dgs is zeroed first, so any previous content will be lost. 
	 * \param[in] ied Face id in face data structure intfac
	 * \param[in] ins Interior state of conserved variables
	 * \param[in|out] gs Ghost state of conserved variables
	 * \param[in|out] dgs Derivatives of ghost state of conserved variables w.r.t.
	 *   the interior state ins, NVARS x NVARS stored in a row-major 1D array
	 */
	void compute_boundary_Jacobian(const int ied, const a_real *const ins, 
			a_real *const gs, a_real *const dgs) const;

	/// Computes viscous flux across a face
	/** The output vflux still needs to be integrated on the face.
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

	/// Computes the spectral radius of the thin-layer Jacobian times the identity matrix
	/** The inputs are same as \ref computeViscousFluxJacobian.
	 */
	void computeViscousFluxApproximateJacobian(const a_int iface,
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
	/** 
	 * \param[in,out] u Vector to store the initial data in
	 */
	void initializeUnknowns(MVector& u) const;
	
	/// Compute nodal quantities to export
	/** \param vec Dummy argument, not used
	 */
	void postprocess_point(const MVector& u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& vec) const;
	
	virtual void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const = 0;
	
	virtual void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A) const = 0;
	
	virtual void getGradients(const MVector& u, MVector grad[NDIM]) const = 0;
	
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
	DiffusionMA(const UMesh2dh *const mesh,            ///< Mesh context
			const a_real diffcoeff,                    ///< Diffusion coefficient 
			const a_real bvalue,                       ///< Constant boundary value
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
				> source,                              ///< Function defining the source term
			std::string grad_scheme                    ///< A string identifying the gradient
			                                           ///< scheme to use
			);
	
	void compute_residual(const MVector& u, MVector& __restrict residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict dtm) const;
	
	/*void add_source(const MVector& u, 
			MVector& __restrict residual, amat::Array2d<a_real>& __restrict dtm) const;*/
	
	void compute_jacobian(const MVector& u, 
			LinearOperator<a_real,a_int> *const A) const;
	
	void getGradients(const MVector& u, MVector grad[NDIM]) const;

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
	
	const GradientScheme *const gradcomp;
};

/// Creates a first-order `thin-layer' Laplacian smoothing matrix
/** \cite{jameson1986}
 */
template <short nvars>
void setupLaplacianSmoothingMatrix(const UMesh2dh *const m, LinearOperator<a_real,a_int> *const M);

}	// end namespace
#endif

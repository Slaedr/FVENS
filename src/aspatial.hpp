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

	/// Faces' Gauss points' coords, stored a 3D array of dimensions naface x nguass x ndim (in that order)
	amat::Array2d<a_real>* gr;
	
	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint();

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face();

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
	virtual void compute_residual(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict__ dtm) = 0;
	
	/// Computes the Jacobian matrix of the residual in DLU format
	virtual void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
			Matrix<a_real,nvars,nvars,RowMajor> *const D, Matrix<a_real,nvars,nvars,RowMajor> *const L, Matrix<a_real,nvars,nvars,RowMajor> *const U) = 0;

	/// Computes the Frechet derivative of the residual along a given direction using finite difference
	/** \param[in] resu The residual vector at the state at which the derivative is to be computed
	 * \param[in] u The state at which the derivative is to be computed
	 * \param[in] v The direction in which the derivative is to be computed
	 * \param[in] add_time_deriv Whether or not time-derivative term is to be added
	 * \param[in] dtm Vector of local time steps for time derivative term
	 * \param[out] prod The vector containing the directional derivative
	 */
	virtual void compute_jac_vec(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod) = 0;
	
	/// Computes a([M du/dt +] dR/du) v + b w and stores in prod
	virtual void compute_jac_gemv(const a_real a, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& w,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod) = 0;
};
	
/// A driver class to control the explicit time-stepping solution using TVD Runge-Kutta time integration
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class EulerFV : public Spatial<NVARS>
{
protected:
	amat::Array2d<a_real> uinf;				///< Free-stream/reference condition
	a_real g;								///< adiabatic index

	/// stores (for each cell i) \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face of the cell
	amat::Array2d<a_real> integ;
	
	/// Analytical flux vector computation
	EulerFlux aflux;
	
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

	/// Limiter context
	FaceDataComputation* lim;
	
	/// Ghost cell flow quantities
	amat::Array2d<a_real> ug;

	amat::Array2d<a_real> dudx;				///< X-gradients at cell centres
	amat::Array2d<a_real> dudy;				///< Y-gradients at cell centres

	int solid_wall_id;						///< Boundary marker corresponding to solid wall
	int inflow_outflow_id;					///< Boundary marker corresponding to inflow/outflow
	int supersonic_vortex_case_inflow;		///< Inflow boundary marker for supersonic vortex case
	
	amat::Array2d<a_real> scalars;			///< Holds density, Mach number and pressure for each cell
	amat::Array2d<a_real> velocities;		///< Holds velocity components for each cell
	
	amat::Array2d<a_real> uleft;			///< Left state at faces
	amat::Array2d<a_real> uright;			///< Right state at faces

	bool matrix_free_implicit;						///< True if matrix-free implicit time-stepping is being used
	Matrix<a_real,Dynamic,Dynamic,RowMajor> aux;	///< Temporary storage needed for matrix free
	a_real eps;										///< step length for finite difference Jacobian

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Array2d<a_real>& instates, amat::Array2d<a_real>& bounstates);

	/// Computes ghost cell state across the face denoted by the first parameter
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs);

public:
	EulerFV(const UMesh2dh *const mesh, std::string invflux, std::string jacflux, std::string reconst, std::string limiter, const bool matrixfree_implicit);
	
	~EulerFV();
	
	/// Set simulation data and precompute data needed for reconstruction
	void loaddata(const short inittype, a_real Minf, a_real vinf, a_real a, a_real rhoinf, Matrix<a_real,Dynamic,Dynamic,RowMajor>& u);

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** This invokes flux calculation after zeroing the residuals and also computes local time steps.
	 */
	void compute_residual(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict__ dtm);

#if HAVE_PETSC==1
	/// Computes the residual Jacobian as a PETSc martrix
	void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, const bool blocked, Mat A);
#else
	/// Computes the residual Jacobian as arrays of diagonal blocks for each cell, and lower and upper blocks for each face
	/** \note D, L and U are not zeroed before use.
	 */
	void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
			Matrix<a_real,NVARS,NVARS,RowMajor> *const D, Matrix<a_real,NVARS,NVARS,RowMajor> *const L, Matrix<a_real,NVARS,NVARS,RowMajor> *const U);

	/// Computes first order directional derivative using real-step finite difference
	void compute_jac_vec(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v, 
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
	
	void compute_jac_gemv(const a_real a, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ w,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
#endif

	/// Compute cell-centred quantities to export
	void postprocess_cell(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, amat::Array2d<a_real>& scalars, amat::Array2d<a_real>& velocities);
	
	/// Compute nodal quantities to export, based on area-weighted averaging (which takes into account ghost cells as well)
	void postprocess_point(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, amat::Array2d<a_real>& scalars, amat::Array2d<a_real>& velocities);

	/// Compute norm of cell-centered entropy production
	/// Call aftr computing pressure etc \sa postprocess_cell
	a_real compute_entropy_cell(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u);
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
	std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> source;
	//void (*const source)(const a_real *const r, const a_real t, const a_real *const u, a_real *const sourceterm);
	std::vector<a_real> h;			///< Size of cells
	
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs);
	
	void compute_boundary_states(const amat::Array2d<a_real>& instates, amat::Array2d<a_real>& bounstates);

public:
	Diffusion(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> source);
	
	virtual void compute_residual(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, 
			amat::Array2d<a_real>& __restrict__ dtm) = 0;
	
	virtual void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
			Matrix<a_real,nvars,nvars,RowMajor> *const D, Matrix<a_real,nvars,nvars,RowMajor> *const L, Matrix<a_real,nvars,nvars,RowMajor> *const U) = 0;

	virtual void compute_jac_vec(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v, 
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod) = 0;
	
	void compute_jac_gemv(const a_real a, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& w,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
	
	virtual void postprocess_point(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, amat::Array2d<a_real>& up);

	virtual ~Diffusion();
};

/// Spatial discretization of diffusion operator with constant difusivity using the thin-layer model
template <short nvars>
class DiffusionThinLayer : public Diffusion<nvars>
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

	amat::Array2d<a_real> uleft;			///< Left state at each face
	amat::Array2d<a_real> ug;				///< Boundary states

public:
	DiffusionThinLayer(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> source);
	
	void compute_residual(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, 
			const bool gettimesteps, amat::Array2d<a_real>& __restrict__ dtm);
	
	void add_source(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, amat::Array2d<a_real>& __restrict__ dtm);
	
	void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
			Matrix<a_real,nvars,nvars,RowMajor> *const D, Matrix<a_real,nvars,nvars,RowMajor> *const L, Matrix<a_real,nvars,nvars,RowMajor> *const U);

	void compute_jac_vec(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v, 
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
	
	void compute_jac_gemv(const a_real a, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& w,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
	
	using Diffusion<nvars>::postprocess_point;
};

/// Spatial discretization of diffusion operator with constant diffusivity using `modified gradient' or `corrected gradient' method
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
			std::function<void(const a_real *const, const a_real, const a_real *const, a_real *const)> source, std::string reconst);
	
	void compute_residual(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, 
			amat::Array2d<a_real>& __restrict__ dtm);
	
	void add_source(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ residual, amat::Array2d<a_real>& __restrict__ dtm);
	
	void compute_jacobian(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& u, 
			Matrix<a_real,nvars,nvars,RowMajor> *const D, Matrix<a_real,nvars,nvars,RowMajor> *const L, Matrix<a_real,nvars,nvars,RowMajor> *const U);

	void compute_jac_vec(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v, 
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);
	
	void compute_jac_gemv(const a_real a, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ resu, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ v,
			const bool add_time_deriv, const amat::Array2d<a_real>& dtm,
			const a_real b, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& w,
			const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ prod);

	~DiffusionMA();
	
	using Diffusion<nvars>::postprocess_point;
};

}	// end namespace
#endif

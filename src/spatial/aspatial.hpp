/** @file aspatial.hpp
 * @brief Spatial discretization for Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016; modified May 13 2017
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ASPATIAL_H
#define ASPATIAL_H 1

#include <array>

#include "aconstants.hpp"

#include "utilities/aarray2d.hpp"

#include "mesh/amesh2dh.hpp"
#include "anumericalflux.hpp"
#include "agradientschemes.hpp"
#include "areconstruction.hpp"
#include "abc.hpp"

#include <petscmat.h>

namespace fvens {

/// Base class for finite volume spatial discretization
template<int nvars>
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
	/** By convention, we need to compute the negative of the nonlinear function whose root
	 * we want to find. For pseudo time-stepping, the output should be -r(u), where the ODE is
	 * M du/dt + r(u) = 0.
	 * \param[in] u The state at which the residual is to be computed
	 * \param[in|out] residual The residual is added to this
	 * \param[in] gettimesteps Whether time-step computation is required
	 * \param[out] dtm Local time steps are stored in this
	 */
	virtual StatusCode compute_residual(const Vec u, Vec residual, 
			const bool gettimesteps, std::vector<a_real>& dtm) const = 0;
	
	/// Computes the Jacobian matrix of the residual r(u)
	/** It is supposed to compute dr/du when we want to solve [M du/dt +] r(u) = 0.
	 */
	virtual StatusCode compute_jacobian(const Vec u, Mat A) const = 0;

	/// Computes gradients of field variables and stores them in the argument
	virtual void getGradients(const MVector& u,
	                          GradArray<nvars>& grads) const = 0;

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in|out] u Vector to store the initial data in
	 */
	virtual StatusCode initializeUnknowns(Vec u) const = 0;
	
	/// Compute nodal quantities to export
	virtual StatusCode postprocess_point(const Vec u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& vector) const = 0;

	/// Exposes access to the mesh context
	const UMesh2dh* mesh() const
	{
		return m;
	}

protected:
	/// Mesh context
	const UMesh2dh *const m;

	/// Cell centers of both real cells and ghost cells
	/** The first nelem rows correspond to real cells, 
	 * the next nelem+nbface rows are ghost cell centres, indexed by nelem+iface for face iface.
	 */
	amat::Array2d<a_real> rc;

	/// Faces' Gauss points' coords, stored a 3D array of dimensions 
	/// naface x nguass x ndim (in that order)
	amat::Array2d<a_real>* gr;
	
	/// computes ghost cell centers assuming symmetry about the midpoint of the boundary face
	void compute_ghost_cell_coords_about_midpoint(amat::Array2d<a_real>& rchg);

	/// computes ghost cell centers assuming symmetry about the face
	void compute_ghost_cell_coords_about_face(amat::Array2d<a_real>& rchg);

	/// Computes a unique face gradient from cell-centred gradients using the modified average method
	/** \param iface The \ref intfac index of the face at which the gradient is to be computed
	 * \param ucl The left cell-centred state
	 * \param ucr The right cell-centred state
	 * \param gradl Left cell-centred gradients
	 * \param gradr Right cell-centred gradients
	 * \param[out] grad Face gradients
	 */
	void getFaceGradient_modifiedAverage(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real gradl[NDIM][nvars], const a_real gradr[NDIM][nvars], a_real grad[NDIM][nvars])
		const;

	/// Computes the thin-layer face gradient and its Jacobian w.r.t. the left and right states
	/** The Jacobians are computed w.r.t. whatever variables 
	 * the derivatives dul and dur are computed with respect to.
	 * \param iface The \ref intfac index of the face at which the gradient Jacobian is to be computed
	 * \param ucl The left state
	 * \param ucr The right state
	 * \param dul The Jacobian of the left state w.r.t. the cell-centred conserved variables
	 * \param dur The Jacobian of the right state w.r.t. the cell-centred conserved variables
	 * \param[out] grad Face gradients
	 * \param[out] dgradl Jacobian of left cell-centred gradients
	 * \param[out] dgradr Jacobian of right cell-centred gradients
	 */
	void getFaceGradientAndJacobian_thinLayer(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real *const dul, const a_real *const dur,
		a_real grad[NDIM][nvars], a_real dgradl[NDIM][nvars][nvars], a_real dgradr[NDIM][nvars][nvars])
		const;
};

/// The collection of physical data needed to initialize flow spatial discretizations
struct FlowPhysicsConfig
{
	a_real gamma;                       ///< Adiabatic index 
	a_real Minf;                        ///< Free-stream Mach number
	a_real Tinf;                        ///< Free-stream temperature in Kelvin
	a_real Reinf;                       ///< Free-stream Reynolds number
	a_real Pr;                          ///< (Constant) Prandtl number
	a_real aoa;                         ///< Angle of attack in radians 
	bool viscous_sim;                   ///< Whether to include viscous effects
	bool const_visc;                    ///< Whether to use constant viscosity
	FlowBCConfig bcconf;                ///< Boundary condition specification
};

/// Collection of options related to the spatial discretization scheme
struct FlowNumericsConfig
{
	std::string conv_numflux;         ///< Convective numerical flux to use
	std::string conv_numflux_jac;     ///< Conv. numer. flux to use for approximate Jacobian
	std::string gradientscheme;       ///< Method to use to compute gradients
	std::string reconstruction;       ///< Method to use to reconstruct the solution
	a_real limiter_param;             ///< Parameter that is required for some limiters
	bool order2;                      ///< Whether to compute a second-order solution
};

/// Abstract base class for finite volume discretization of flow problems
/** This is meant to be a template-parameter-free abstract class encapsulating a spatial discretization
 * for a flow problem.
 */
class FlowFV_base : public Spatial<NVARS>
{
public:
	/// Sets data and initializes the numerics
	FlowFV_base(const UMesh2dh *const mesh,              ///< Mesh context
	            const FlowPhysicsConfig& pconfig,        ///< Physical data defining the problem
	            const FlowNumericsConfig& nconfig        ///< Options defining the numerical method
	            );

	virtual ~FlowFV_base();

	/// Sets initial conditions
	/** \param[in] fromfile True if initial data is to be read from a file
	 * \param[in] file Name of initial conditions file
	 * \param[in,out] u Vector to store the initial data in
	 */
	StatusCode initializeUnknowns(Vec u) const;

	/// Compute cell-centred quantities to export \deprecated Use postprocess_point instead.
	StatusCode postprocess_cell(const Vec u, amat::Array2d<a_real>& scalars, 
	                            amat::Array2d<a_real>& velocities) const;
	
	/// Compute nodal quantities to export
	/** Based on area-weighted averaging which takes into account ghost cells as well.
	 * Density, Mach number, pressure and temperature are the exported scalars,
	 * and velocity is exported as well.
	 */
	StatusCode postprocess_point(const Vec u, amat::Array2d<a_real>& scalars, 
	                             amat::Array2d<a_real>& velocities) const;

	/// Compute norm of cell-centered entropy production
	/** Call after computing pressure etc \sa postprocess_cell
	 */
	a_real compute_entropy_cell(const Vec u) const;

	/// Computes gradients of converved variables
	void getGradients(const MVector& u, GradArray<NVARS>& grads) const;

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
	const GradientScheme<NVARS> *const gradcomp;

	/// Reconstruction context
	const SolutionReconstruction *const lim;

	/// The different boundary conditions required for all the boundaries
	const std::map<int,const FlowBC<a_real>*> bcs;

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
	 * \param[in,out] gs Ghost state of conserved variables
	 */
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const gs) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	/** The output array dgs is zeroed first, so any previous content will be lost. 
	 * \param[in] ied Face id in face data structure intfac
	 * \param[in] ins Interior state of conserved variables
	 * \param[in,out] gs Ghost state of conserved variables
	 * \param[in,out] dgs Derivatives of ghost state of conserved variables w.r.t.
	 *   the interior state ins, NVARS x NVARS stored in a row-major 1D array
	 */
	void compute_boundary_Jacobian(const int ied, const a_real *const ins, 
			a_real *const gs, a_real *const dgs) const;
};

/// Computes the integrated fluxes and their Jacobians for compressible flow
/** Note about BCs: normal velocity is assumed zero at all walls.
 * \note Make sure compute_topological(), compute_face_data() and compute_jacobians() 
 * have been called on the mesh object prior to initialzing an object of this class.
 */
template <
	bool secondOrderRequested,      ///< Whether to computes gradients to get a 2nd order solution
	bool constVisc                  ///< Whether to use constant viscosity (true) or Sutherland (false)
>
class FlowFV : public FlowFV_base
{
public:
	/// Sets data and initializes the numerics
	FlowFV(const UMesh2dh *const mesh,                  ///< Mesh context
		const FlowPhysicsConfig& pconfiguration,        ///< Physical data defining the problem
		const FlowNumericsConfig& nconfiguration        ///< Options defining the numerical method
	);
	
	~FlowFV();

	/// Calls functions to assemble the [right hand side](@ref residual)
	/** Actually computes -r(u) (ie., negative of r(u)), where the nonlinear problem being solved is 
	 * [M du/dt +] r(u) = 0. 
	 * Invokes flux calculation and adds the fluxes to the residual vector,
	 * and also computes local time steps.
	 */
	StatusCode compute_residual(const Vec u, Vec residual, 
			const bool gettimesteps, std::vector<a_real>& dtm) const;

	/// Computes the residual Jacobian as a PETSc martrix
	/** Computes the Jacobian of r(u), where the 
	 */
	StatusCode compute_jacobian(const Vec u, Mat A) const;
	
protected:
	/// Computes viscous flux across a face
	/** The output vflux still needs to be integrated on the face.
	 * \param[in] iface Face index
	 * \param[in] ucell_l Cell-centred conserved variables on left side of the face
	 * \param[in] ucell_r Cell-centred conserved variables on right side of the face
	 *             Note that for boundary faces, this can be NULL because ug is used instead.
	 * \param[in] ug Ghost cell-centred conserved variables
	 * \param[in] dudx Cell-centred gradients ("optional")
	 * \param[in] dudy Cell-centred gradients ("optional", see below)
	 * \param[in] ul Left state of faces (conserved variables)
	 * \param[in] ul Right state of faces (conserved variables)
	 * \param[in,out] vflux On output, contains the viscous flux across the face
	 *
	 * Note that dudx and dudy can be unallocated if only first-order fluxes are being computed,
	 * but ul and ur are always used.
	 */
	void computeViscousFlux(
			const a_int iface, const a_real *const ucell_l, const a_real *const ucell_r,
			const amat::Array2d<a_real>& ug,
			const GradArray<NVARS>& grads,
			const amat::Array2d<a_real>& ul, const amat::Array2d<a_real>& ur,
			a_real *const vflux) const;

	/// Compues the first-order "thin-layer" viscous flux Jacobian
	/** This is the same sign as is needed in the residual; note that the viscous flux Jacobian is
	 * added to the output matrices - they are not zeroed or directly assigned to.
	 * The outputs vfluxi and vfluxj still need to be integrated on the face.
	 * \param[in] iface Face index
	 * \param[in] ul Cell-centred conserved variable on left
	 * \param[in] ur Cell-centred conserved variable on right
	 * \param[in,out] vfluxi Flux Jacobian \f$ \partial \mathbf{f}_{ij} / \partial \mathbf{u}_i \f$
	 *   NVARS x NVARS array stored as a 1D row-major array
	 * \param[in,out] vfluxj Flux Jacobian \f$ \partial \mathbf{f}_{ij} / \partial \mathbf{u}_j \f$
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
template <int nvars>
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
	StatusCode initializeUnknowns(Vec u) const;
	
	/// Compute nodal quantities to export
	/** \param vec Dummy argument, not used
	 */
	StatusCode postprocess_point(const Vec u, amat::Array2d<a_real>& scalars, 
			amat::Array2d<a_real>& vec) const;
	
	virtual StatusCode compute_residual(const Vec u, Vec residual, 
			const bool gettimesteps, std::vector<a_real>& dtm) const = 0;
	
	virtual StatusCode compute_jacobian(const Vec u, Mat A) const = 0;
	
	virtual void getGradients(const MVector& u,
	                          GradArray<nvars>& grads) const = 0;
	
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

	/// Dirichlet BC for a boundary face ied
	void compute_boundary_state(const int ied, const a_real *const ins, a_real *const bs) const;
	
	/// Dirichlet BC for all boundaries
	void compute_boundary_states(const amat::Array2d<a_real>& instates, 
			amat::Array2d<a_real>& bounstates) const;
};

/// Spatial discretization of diffusion operator with constant diffusivity 
/// using `modified gradient' or `corrected gradient' method
template <int nvars>
class DiffusionMA : public Diffusion<nvars>
{
public:
	DiffusionMA(const UMesh2dh *const mesh,            ///< Mesh context
			const a_real diffcoeff,                    ///< Diffusion coefficient 
			const a_real bvalue,                       ///< Constant boundary value
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
				> source,                              ///< Function defining the source term
			const std::string grad_scheme              ///< A string identifying the gradient
			                                           ///< scheme to use
			);
	
	StatusCode compute_residual(const Vec u, Vec residual, 
							const bool gettimesteps, std::vector<a_real>& dtm) const;
	
	/*void add_source(const MVector& u, 
			MVector& __restrict residual, amat::Array2d<a_real>& __restrict dtm) const;*/
	
	StatusCode compute_jacobian(const Vec u, Mat A) const;
	
	void getGradients(const MVector& u,
	                  GradArray<nvars>& grads) const;

	~DiffusionMA();

protected:
	using Diffusion<nvars>::postprocess_point;
	using Spatial<nvars>::m;
	using Spatial<nvars>::rc;
	using Spatial<nvars>::gr;
	using Spatial<nvars>::getFaceGradient_modifiedAverage;
	using Spatial<nvars>::getFaceGradientAndJacobian_thinLayer;
	using Diffusion<nvars>::diffusivity;
	using Diffusion<nvars>::bval;
	using Diffusion<nvars>::source;
	using Diffusion<nvars>::h;
	
	using Diffusion<nvars>::compute_boundary_state;
	using Diffusion<nvars>::compute_boundary_states;
	
	const GradientScheme<nvars> *const gradcomp;
};

}	// end namespace
#endif

/** @file
 * @brief Spatial discretization for Euler/Navier-Stokes equations.
 * @author Aditya Kashi
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

#ifndef FVENS_FLOW_SPATIAL_H
#define FVENS_FLOW_SPATIAL_H

#include "aspatial.hpp"
#include "anumericalflux.hpp"
#include "agradientschemes.hpp"
#include "areconstruction.hpp"
#include "abc.hpp"

namespace fvens {

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
	std::vector<FlowBCConfig> bcconf;   ///< Boundary condition specification
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
template <typename scalar>
class FlowFV_base : public Spatial<scalar,NVARS>
{
public:
	/// Sets data and initializes the numerics
	FlowFV_base(const UMesh2dh<scalar> *const mesh,              ///< Mesh context
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

	/// Computes the residual from a PETSc vector containing the conserved variables
	/** \sa compute_residual
	 * Uses \ref compute_residual from derived classes to asseble the residual and compute max
	 * allowable time steps.
	 *
	 * \warning Data from the PETSc vector is extracted as a pointer to PetscScalar, PETSc's native
	 *   floating point type. This may not be the template parameter to this class.
	 *
	 * \note For non-standard scalar types (like std::complex or ADOL-C's adouble), this function
	 *   needs to be reimplementated in an appropriate sub-class.
	 */
	virtual StatusCode assemble_residual(const Vec u, Vec residual, 
	                                     const bool gettimesteps,
	                                     std::vector<a_real>& dtm) const;

	/// Computes the [right hand side](@ref residual)
	/** Actually computes -r(u) (ie., negative of r(u)), where the nonlinear problem being solved is 
	 * [M du/dt +] r(u) = 0. 
	 * Invokes flux calculation and adds the fluxes to the residual vector,
	 * and also computes local time steps.
	 */
	virtual StatusCode compute_residual(const scalar *const u, scalar *const __restrict residual,
	                                    const bool gettimesteps, std::vector<a_real>& dtm) const = 0;

	/// Computes Cp, Csf, Cl, Cd_p and Cd_sf on one surface
	/** \param[in] u The multi-vector containing conserved variables
	 * \param[in] grad Gradients of converved variables at cell-centres
	 * \param[in] iwbcm The marker of the boundary on which the computation is to be done
	 * \param[in,out] output On output, contains for each boundary face having the marker im : 
	 *                   Cp and Csf, in that order
	 * \return A tuple containing Cl, Cd_p and Cd_sf.
	 *
	 * \todo Write unit tests
	 * \todo Generalize to 3D
	 */
	std::tuple<scalar,scalar,scalar> computeSurfaceData(const MVector<scalar>& u,
	                                                    const GradArray<scalar,NVARS>& grad,
	                                                    const int iwbcm,
	                                                    MVector<scalar>& output) const;

	/// Computes gradients of converved variables
	void getGradients(const MVector<scalar>& u, GradArray<scalar,NVARS>& grads) const;

protected:

	using Spatial<scalar,NVARS>::m;
	using Spatial<scalar,NVARS>::rc;
	using Spatial<scalar,NVARS>::gr;
	using Spatial<scalar,NVARS>::getFaceGradient_modifiedAverage;

	/// Problem specification
	const FlowPhysicsConfig& pconfig;

	/// Numerical method specification
	const FlowNumericsConfig& nconfig;

	/// Analytical flux vector computation
	const IdealGasPhysics<scalar> physics;

	/// Free-stream/reference condition
	const std::array<scalar,NVARS> uinf;

	/// Numerical inviscid flux calculation context for residual computation
	/** This is the "actual" flux being used.
	 */
	const InviscidFlux<scalar> *const inviflux;

	/// Gradient computation context
	const GradientScheme<scalar,NVARS> *const gradcomp;

	/// Reconstruction context
	const SolutionReconstruction<scalar,NVARS> *const lim;

	/// The different boundary conditions required for all the boundaries
	const std::map<int,const FlowBC<scalar>*> bcs;

	/// Computes flow variables at all boundaries (either Gauss points or ghost cell centers) 
	/// using the interior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Array2d<scalar>& instates, 
			amat::Array2d<scalar>& bounstates) const;

	/// Computes ghost cell state across one face
	/** \param[in] ied Face id in face data structure intfac
	 * \param[in] ins Interior state of conserved variables
	 * \param[in,out] gs Ghost state of conserved variables
	 */
	void compute_boundary_state(const int ied, const scalar *const ins, scalar *const gs) const;
};

/// Computes the integrated fluxes and their Jacobians for compressible flow
/** \note Make sure [mesh proprocessing](\ref preprocessMesh) has been completed
 * prior to initialzing an object of this class.
 * 
 * This class also includes (as protected members) clones of some objects from the base class prefixed
 * with 'j' (such as \ref jphy). These objects are used in the computation of approximate Jacobians
 * and only use the basic real type, not the generic scalar used for computing the fluxes. These
 * objects, except the one for numerical inviscid flux \ref jflux, are meant to be logically
 * equivalent to the corresponding objects for the fluxes in the base flow class with only a change in
 * the scalar type.
 */
template <
	typename scalar,
	bool secondOrderRequested,      ///< Whether to compute gradients to get a 2nd order solution
	bool constVisc                  ///< Whether to use constant viscosity (true) or Sutherland (false)
>
class FlowFV : public FlowFV_base<scalar>
{
public:
	/// Sets data and initializes the numerics
	FlowFV(const UMesh2dh<scalar> *const mesh,                  ///< Mesh context
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
	StatusCode compute_residual(const scalar *const u, scalar *const residual,
	                            const bool gettimesteps, std::vector<a_real>& dtm) const;

	/// Computes the residual Jacobian as a PETSc martrix
	/** Computes the Jacobian of r(u), where the 
	 */
	virtual StatusCode compute_jacobian(const Vec u, Mat A) const;
	
protected:
	using Spatial<scalar,NVARS>::m;
	using Spatial<scalar,NVARS>::rc;
	using Spatial<scalar,NVARS>::gr;
	using Spatial<scalar,NVARS>::getFaceGradient_modifiedAverage;
	using Spatial<scalar,NVARS>::getFaceGradientAndJacobian_thinLayer;
	using FlowFV_base<scalar>::pconfig;
	using FlowFV_base<scalar>::nconfig;
	using FlowFV_base<scalar>::physics;
	using FlowFV_base<scalar>::inviflux;
	using FlowFV_base<scalar>::gradcomp;
	using FlowFV_base<scalar>::lim;
	using FlowFV_base<scalar>::bcs;
	using FlowFV_base<scalar>::compute_boundary_states;

	/// Gas physics to use for computing analytical Jacobian
	/** This should usually be same as \ref physics used for the flux computation. This has been
	 * provided so that analytical Jacobians can be constructed when the scalar is not double or float,
	 * if need be.
	 */
	const IdealGasPhysics<a_real> jphy;

	/// Free-stream/reference condition for Jacobian
	const std::array<a_real,NVARS> juinf;

	/// Numerical inviscid flux context for computing an analytical Jacobian
	/** See \ref jphy for the reason this is provided.
	 * This may implement a numerical flux function mathematically distint from the
	 * \ref FlowFV_base::inviflux object used for the fluxes (right hand side).
	 */
	const InviscidFlux<a_real> *const jflux;

	/// Boundary conditions objects used for building the Jacobian
	const std::map<int,const FlowBC<a_real>*> jbcs;

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
			const a_int iface, const scalar *const ucell_l, const scalar *const ucell_r,
			const amat::Array2d<scalar>& ug,
			const GradArray<scalar,NVARS>& grads,
			const amat::Array2d<scalar>& ul, const amat::Array2d<scalar>& ur,
			scalar *const vflux) const;

	/// Compues the first-order "thin-layer" viscous flux Jacobian
	/** This is the same sign as is needed in the residual; note that the viscous flux Jacobian is
	 * added to the output matrices - the latter are not zeroed nor directly assigned to.
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

}

#endif

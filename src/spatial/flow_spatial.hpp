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

	/// Computes the residual and local time steps
	/** By convention, we need to compute the negative of the nonlinear function whose root
	 * we want to find. Note that our nonlinear function or residual is defined (for steady problems) as
	 * the sum (over all cells) of net outgoing fluxes from each cell
	 * \f$ r(u) = \sum_K \int_{\partial K} F \hat{n} d\gamma \f$ where $f K $f denotes a cell.
	 * For pseudo time-stepping, the output should be -r(u), where the ODE is
	 * M du/dt + r(u) = 0.
	 *
	 * \param[in] u The state at which the residual is to be computed
	 * \param[in|out] residual The residual is added to this
	 * \param[in] gettimesteps Whether time-step computation is required
	 * \param[in,out] timesteps Local time steps are stored in this (shoud be pre-allocated)
	 */
	virtual StatusCode compute_residual(const Vec u, Vec residual,
	                                    const bool gettimesteps, Vec timesteps) const = 0;

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
	std::tuple<scalar,scalar,scalar>
	computeSurfaceData(const MVector<scalar>& u,
	                   const GradBlock_t<scalar,NDIM,NVARS> *const grad,
	                   const int iwbcm,
	                   MVector<scalar>& output) const;

	/// Computes gradients of converved variables
	void getGradients(const MVector<scalar>& u, GradBlock_t<scalar,NDIM,NVARS> *const grads) const;

protected:

	using Spatial<scalar,NVARS>::m;
	using Spatial<scalar,NVARS>::rcvec;
	using Spatial<scalar,NVARS>::rch;
	using Spatial<a_real,NVARS>::rcbp;
	using Spatial<scalar,NVARS>::gr;
	using Spatial<scalar,NVARS>::getFaceGradient_modifiedAverage;

	/// Problem specification
	const FlowPhysicsConfig pconfig;

	/// Numerical method specification
	const FlowNumericsConfig nconfig;

	/// Analytical flux vector computation
	const IdealGasPhysics<scalar> physics;

	/// Free-stream/reference condition
	const std::array<a_real,NVARS> uinf;

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
	void compute_boundary_states(const scalar *const instates, scalar *const ghoststates) const;

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
	StatusCode compute_residual(const Vec u, Vec residual,
	                            const bool gettimesteps, Vec timesteps) const;

	/// Computes fluxes into the residual vector
	void compute_fluxes(const scalar *const u, const scalar *const gradients,
	                    const scalar *const uleft, const scalar *const uright,
	                    const scalar *const ug,
	                    scalar *const res) const;

	/// Computes the maximum allowable time step at each cell
	/** This is the volume of the cell divided by the integral over the cell boundary of
	 * the spectral radius of the analytical flux Jacobian.
	 */
	void compute_max_timestep(const amat::Array2dView<scalar> uleft,
	                          const amat::Array2dView<scalar> uright,
	                          a_real *const timesteps) const;

	/// Computes the blocks of the Jacobian matrix for the flux across an interior face
	/** \see Spatial::compute_local_jacobian_interior
	 */
	void compute_local_jacobian_interior(const a_int iface,
	                                     const a_real *const ul, const a_real *const ur,
	                                     Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& L,
	                                     Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& U) const;

	/// Computes the blocks of the Jacobian matrix for the flux across a boundary face
	/** \see Spatial::compute_local_jacobian_boundary
	 */
	void compute_local_jacobian_boundary(const a_int iface,
	                                     const a_real *const ul,
	                                     Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& L) const;

protected:
	using Spatial<scalar,NVARS>::m;
	using Spatial<scalar,NVARS>::rcvec;
	using Spatial<scalar,NVARS>::rch;
	using Spatial<a_real,NVARS>::rcbp;
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

	/// Computes viscous flux across a face at one point
	/** The output vflux still needs to be integrated on the face.
	 * \param[in] normals
	 * \param[in] ccleft
	 * \param[in] ccright
	 * \param[in] ucell_l Cell-centred conserved variables on left side of the face
	 * \param[in] ucell_r Cell-centred conserved variables on right side of the face
	 * \param[in] gradsLeft Cell-centred gradients ("optional", see below)
	 * \param[in] gradsRight Cell-centred gradients ("optional", see below)
	 * \param[in] ul Left state of faces (conserved variables)
	 * \param[in] ur Right state of faces (conserved variables)
	 * \param[in,out] vflux On output, contains the viscous flux across the face
	 *
	 * Note that grads can be unallocated if only first-order fluxes are being computed,
	 * but ul and ur are always used.
	 */
	void compute_viscous_flux(const scalar *const normal,
	                          const scalar *const ccleft, const scalar *const ccright,
	                          const scalar *const ucell_l, const scalar *const ucell_r,
	                          const GradBlock_t<scalar,NDIM,NVARS>& gradsLeft,
	                          const GradBlock_t<scalar,NDIM,NVARS>& gradsRight,
	                          const scalar *const ul, const scalar *const ur,
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
	void compute_viscous_flux_jacobian(const a_int iface,
	                                   const a_real *const ul, const a_real *const ur,
	                                   a_real *const __restrict vfluxi,
	                                   a_real *const __restrict vfluxj) const;

	/// Computes the spectral radius of the thin-layer Jacobian times the identity matrix
	/** The inputs are same as \ref compute_viscous_flux_jacobian
	 */
	void compute_viscous_flux_approximate_jacobian(const a_int iface,
	                                               const a_real *const ul, const a_real *const ur,
	                                               a_real *const __restrict vfluxi,
	                                               a_real *const __restrict vfluxj) const;
};

}

#endif

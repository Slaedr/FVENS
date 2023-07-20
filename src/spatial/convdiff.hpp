/** \file difusion.hpp
 * \brief Discretization for scalar or decoupled vector diffusion equation
 */

#ifndef FVENS_SPATIAL_CONVECTION_DIFFUSION_H
#define FVENS_SPATIAL_CONVECTION_DIFFUSION_H

#include <vector>
#include <functional>
#include <petscvec.h>

#include "../linalg/tracevector.hpp"
#include "agradientschemes.hpp"
#include "areconstruction.hpp"
#include "aspatial.hpp"
#include "bc_convdiff.hpp"

namespace fvens {
namespace convdiff {


/// Physical parameters of the convection diffusion problem
struct PhysicsConfig {
	std::array<freal, NDIM> conv_velocity;      ///< Dimensionless convection velocity
	bool viscous_sim;                           ///< Whether there's diffusion enabled
	freal peclet;                               ///< Peclet number
	/// Function defining the source term
	/** First parameter is the spatial location, second parameter is any scalar input,
	 * third parameter is the state vector at that location, and the fourth is the
	 * output variable for the source term value(s) at that location.
	 */
	std::function <void(const freal *, freal, const freal *, freal *)> source;
	std::vector<BCConfig> bcconf;               ///< Boundary condition specification
};

/// Collection of options related to the spatial discretization scheme
struct NumericsConfig
{
	std::string gradientscheme;       ///< Method to use to compute gradients
	std::string reconstruction;       ///< Method to use to reconstruct the solution
	freal limiter_param;              ///< Parameter that is required for some limiters
	bool order2;                      ///< Whether to compute a second-order solution
};

/// Spatial discretization of linear convection-diffusion operator
/** Assumes constant diffusivity and convection speed.
 *  Uses `modified gradient' or `corrected gradient' method for viscous fluxes
 *  and upwind convection.
 */
template <int nvars>
class ConvectionDiffusion : public Spatial<freal,nvars>
{
public:
	/** Set the parameters and set up the required objects.
	 *
	 * @param mesh  The mesh context
	 * @param pconfig  Physical simulation parameters.
	 * @param nconfig  Parameters for the numerical methods to use.
	 */
	ConvectionDiffusion(const UMesh<freal,NDIM> *const mesh,
						const PhysicsConfig& pconfig, const NumericsConfig& nconfig);
	
	StatusCode compute_residual(const Vec u, Vec residual,
	                            const bool gettimesteps, Vec timesteps) const;

	void compute_local_jacobian_interior(const fint iface,
	                                     const freal *const ul, const freal *const ur,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& U) const;

	void compute_local_jacobian_boundary(const fint iface,
	                                     const freal *const ul,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L) const;

	virtual ~ConvectionDiffusion();

protected:
	using Spatial<freal,nvars>::m;
	using Spatial<freal,nvars>::rcvec;
	using Spatial<freal,nvars>::rch;
	using Spatial<freal,nvars>::rcbp;
	using Spatial<freal,nvars>::rcbptr;
	using Spatial<freal,nvars>::gr;
	using Spatial<freal,nvars>::getFaceGradient_modifiedAverage;
	using Spatial<freal,nvars>::getFaceGradientAndJacobian_thinLayer;

	const PhysicsConfig pconfig;
	const NumericsConfig nconfig;

	std::vector<freal> h;			///< Size of cells

	/// Gradient computation method
	std::unique_ptr<const GradientScheme<freal,nvars>> gradcomp;

	/// Reconstruction context
	std::unique_ptr<const SolutionReconstruction<freal,nvars>> lim;

	/// The different boundary conditions required for all the boundaries
	const std::map<int,std::unique_ptr<const BC>> bcs;

	//bool secondOrderRequested = true;

	/// Reconstructed states at all faces
	/** Ideally, this would be local inside compute_residual. However, its setup is non-trivial and
	 * only depends on the mesh, so we do it only once in the constructor and re-use it for all
	 * residual computations.
	 * \note If memory for implicit solves is an issue, this can be moved inside \ref compute_residual
	 *  in order to free up space during the implicit solve.
	 */
	mutable L2TraceVector<freal,nvars> uface;

	/// Gradients at cell centres
	/** This could be local to compute_residual, but same reason as for \ref uface applies here as well.
	 */
	mutable Vec gradvec;

	/// Compute advective flux across one face
	/** In the future, this should be replaced by a call to a numerical flux method depending
	 * on the selected advective physics - linear or nonlinear.
	 */
	void compute_advective_flux(const freal *uleft, const freal *urght,
								const freal *normal,
								freal *__restrict__ fluxes) const;

	/// Compute diffusive flux across one face
    void compute_viscous_flux(const freal *normal, const freal *rcl, const freal *rcr,
                       const freal *ucell_l, const freal *ucell_r,
                       const GradBlock_t<freal,NDIM,nvars>& gradsl,
                       const GradBlock_t<freal,NDIM,nvars>& gradsr,
                       freal *__restrict__ vflux) const;

	void compute_fluxes(const freal *u, const freal *gradients,
                 const freal *uleft, const freal *uright, const freal *ug, freal *res) const;

	void compute_boundary_states(const freal *ins, freal *gs) const;

	void compute_max_timestep(amat::Array2dView<freal> uleft, amat::Array2dView<freal> uright,
							  freal *timesteps) const;
};

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh<freal,NDIM> *const m, const Vec uvec,
                                    amat::Array2d<freal>& up);

}
}

#endif

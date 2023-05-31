/** \file difusion.hpp
 * \brief Discretization for scalar or decoupled vector diffusion equation
 */

#ifndef FVENS_SPATIAL_CONVECTION_DIFFUSION_H
#define FVENS_SPATIAL_CONVECTION_DIFFUSION_H

#include <vector>
#include <functional>
#include <petscvec.h>
#include "agradientschemes.hpp"
#include "diffusion.hpp"

namespace fvens {
namespace convdiff {


/// Types of boundary conditions available for the convection-diffusion problem
enum class BCType {
    dirichlet,
	neumann
};

/// Definition of boundary condition at one particular boundary
/** This is essentially raw data read from the control file.
 */
struct BCConfig
{
	int bc_tag;                   ///< Boundary marker in mesh file
	BCType bc_type;               ///< Type of boundary
	std::vector<freal> bc_vals;   ///< Boundary value(s)
};

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
	freal limiter_param;             ///< Parameter that is required for some limiters
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

	const GradientScheme<freal,nvars> *const gradcomp;

	/// The different boundary conditions required for all the boundaries
	const std::map<int,BCType> bcs;

	bool secondOrderRequested = true;

	/// Reconstructed states at all faces
	/** Ideally, this would be local inside compute_residual. However, its setup is non-trivial and
	 * only depends on the mesh, so we do it only once in the constructor and re-use it for all
	 * residual computations.
	 * \note If memory for implicit solves is an issue, this can be moved inside \ref compute_residual
	 *  in order to free up space during the implicit solve.
	 */
	mutable L2TraceVector<scalar,NVARS> uface;

	void compute_flux_interior(const fint iface,
	                           const amat::Array2dView<freal>& rc,
	                           const freal *const uarr,
	                           const GradBlock_t<freal,NDIM,nvars> *const grads,
	                           amat::Array2dMutableView<freal>& residual) const;

	/// Boundary condition calculation for a single boundary face
	void compute_boundary_state(const int ied, const freal *const ins, freal *const bs) const
	{
		const std::array<scalar,NDIM> n = m->gnormal(ied);
		const auto tag = bcs.at(m->gbtags(ied,0));
		if(tag == BCType::dirichlet) {
			for(int ivar = 0; ivar < nvars; ivar++)
				bs[ivar] = 2.0*bval - ins[ivar];
		} else {
			for(int ivar = 0; ivar < nvars; ivar++)
				bs[ivar] = ins[ivar];
		}
	}
};

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh<freal,NDIM> *const m, const Vec uvec,
                                    amat::Array2d<freal>& up);

}
}

#endif

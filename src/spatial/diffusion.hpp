/** \file difusion.hpp
 * \brief Discretization for scalar or decoupled vector diffusion equation
 */

#ifndef FVENS_DIFFUSION_H
#define FVENS_DIFFUSION_H

#include <vector>
#include <petscvec.h>
#include "agradientschemes.hpp"
#include "aspatial.hpp"

namespace fvens {

/// Spatial discretization of diffusion operator with constant difusivity
template <int nvars>
class Diffusion : public Spatial<freal,nvars>
{
public:
	Diffusion(const UMesh<freal,NDIM> *const mesh, const freal diffcoeff, const freal bvalue,
			std::function <
			void(const freal *const, const freal, const freal *const, freal *const)
			> source);

	virtual void getGradients(const Vec u,
	                          GradBlock_t<freal,NDIM,nvars> *const grads) const = 0;
	
	virtual ~Diffusion();

protected:
	using Spatial<freal,nvars>::m;
	using Spatial<freal,nvars>::rcvec;
	using Spatial<freal,nvars>::rcbp;
	using Spatial<freal,nvars>::gr;
	using Spatial<freal,nvars>::getFaceGradient_modifiedAverage;

	const freal diffusivity;		///< Diffusion coefficient (eg. kinematic viscosity)
	const freal bval;				///< Dirichlet boundary value
	
	/// Pointer to a function that describes the  source term
	std::function <
		void(const freal *const, const freal, const freal *const, freal *const)
					> source;

	std::vector<freal> h;			///< Size of cells

	/// Constant Dirichlet BC for a boundary face ied
	void compute_boundary_state(const int ied, const freal *const ins, freal *const bs) const
	{
		for(int ivar = 0; ivar < nvars; ivar++)
			bs[ivar] = 2.0*bval - ins[ivar];
	}
	
	/// Dirichlet BC for all boundaries
	void compute_boundary_states(const freal *const instates, 
	                             freal *const bounstates) const;
};

/// Spatial discretization of diffusion operator with constant diffusivity 
/// using `modified gradient' or `corrected gradient' method
template <int nvars>
class DiffusionMA : public Diffusion<nvars>
{
public:
	DiffusionMA(const UMesh<freal,NDIM> *const mesh,    ///< Mesh context
			const freal diffcoeff,                    ///< Diffusion coefficient 
			const freal bvalue,                       ///< Constant boundary value
			std::function <
			void(const freal *const, const freal, const freal *const, freal *const)
				> source,                              ///< Function defining the source term
			const std::string grad_scheme              ///< A string identifying the gradient
			                                           ///< scheme to use
			);
	
	StatusCode compute_residual(const Vec u, Vec residual,
	                            const bool gettimesteps, Vec timesteps) const;

	void compute_local_jacobian_interior(const fint iface,
	                                     const freal *const ul, const freal *const ur,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& U) const;

	void compute_local_jacobian_boundary(const fint iface,
	                                     const freal *const ul,
	                                     Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L) const;
	
	void getGradients(const Vec u,
	                  GradBlock_t<freal,NDIM,nvars> *const grads) const;

	~DiffusionMA();

protected:
	using Spatial<freal,nvars>::m;
	using Spatial<freal,nvars>::rcvec;
	using Spatial<freal,nvars>::rch;
	using Spatial<freal,nvars>::rcbp;
	using Spatial<freal,nvars>::rcbptr;
	using Spatial<freal,nvars>::gr;
	using Spatial<freal,nvars>::getFaceGradient_modifiedAverage;
	using Spatial<freal,nvars>::getFaceGradientAndJacobian_thinLayer;

	using Diffusion<nvars>::diffusivity;
	using Diffusion<nvars>::bval;
	using Diffusion<nvars>::source;
	using Diffusion<nvars>::h;

	using Diffusion<nvars>::compute_boundary_state;
	using Diffusion<nvars>::compute_boundary_states;
	
	const GradientScheme<freal,nvars> *const gradcomp;

	void compute_flux_interior(const fint iface,
	                           const amat::Array2dView<freal>& rc,
	                           const freal *const uarr,
	                           const GradBlock_t<freal,NDIM,nvars> *const grads,
	                           amat::Array2dMutableView<freal>& residual) const;
};

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh<freal,NDIM> *const m, const Vec uvec,
                                    amat::Array2d<freal>& up);

}

#endif

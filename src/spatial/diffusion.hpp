/** \file difusion.hpp
 * \brief Discretization for scalar or decoupled vector diffusion equation
 */

#ifndef FVENS_DIFFUSION_H
#define FVENS_DIFFUSION_H

#include <vector>
#include <petscvec.h>
#include "spatial/aspatial.hpp"
#include "agradientschemes.hpp"

namespace fvens {

/// Spatial discretization of diffusion operator with constant difusivity
template <int nvars>
class Diffusion : public Spatial<a_real,nvars>
{
public:
	Diffusion(const UMesh2dh<a_real> *const mesh, const a_real diffcoeff, const a_real bvalue,
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
			> source);

	virtual void getGradients(const MVector<a_real>& u,
	                          GradBlock_t<a_real,NDIM,nvars> *const grads) const = 0;
	
	virtual ~Diffusion();

protected:
	using Spatial<a_real,nvars>::m;
	using Spatial<a_real,nvars>::rc;
	using Spatial<a_real,nvars>::rcbp;
	using Spatial<a_real,nvars>::gr;
	using Spatial<a_real,nvars>::getFaceGradient_modifiedAverage;

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
	void compute_boundary_states(const a_real *const instates, 
	                             a_real *const bounstates) const;
};

/// Spatial discretization of diffusion operator with constant diffusivity 
/// using `modified gradient' or `corrected gradient' method
template <int nvars>
class DiffusionMA : public Diffusion<nvars>
{
public:
	DiffusionMA(const UMesh2dh<a_real> *const mesh,    ///< Mesh context
			const a_real diffcoeff,                    ///< Diffusion coefficient 
			const a_real bvalue,                       ///< Constant boundary value
			std::function <
			void(const a_real *const, const a_real, const a_real *const, a_real *const)
				> source,                              ///< Function defining the source term
			const std::string grad_scheme              ///< A string identifying the gradient
			                                           ///< scheme to use
			);
	
	StatusCode compute_residual(const Vec u, Vec residual,
	                            const bool gettimesteps, Vec timesteps) const;

	void compute_local_jacobian_interior(const a_int iface,
	                                     const a_real *const ul, const a_real *const ur,
	                                     Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& L,
	                                     Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& U) const;

	void compute_local_jacobian_boundary(const a_int iface,
	                                     const a_real *const ul,
	                                     Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& L) const;

	/*void add_source(const MVector<scalar>& u, 
			MVector<scalar>& __restrict residual, amat::Array2d<a_real>& __restrict dtm) const;*/
	
	void getGradients(const MVector<a_real>& u,
	                  GradBlock_t<a_real,NDIM,nvars> *const grads) const;

	~DiffusionMA();

protected:
	using Spatial<a_real,nvars>::m;
	using Spatial<a_real,nvars>::rc;
	using Spatial<a_real,nvars>::rcbp;
	using Spatial<a_real,nvars>::gr;
	using Spatial<a_real,nvars>::getFaceGradient_modifiedAverage;
	using Spatial<a_real,nvars>::getFaceGradientAndJacobian_thinLayer;

	using Diffusion<nvars>::diffusivity;
	using Diffusion<nvars>::bval;
	using Diffusion<nvars>::source;
	using Diffusion<nvars>::h;

	using Diffusion<nvars>::compute_boundary_state;
	using Diffusion<nvars>::compute_boundary_states;
	
	const GradientScheme<a_real,nvars> *const gradcomp;
};

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh2dh<a_real> *const m, const Vec uvec,
                                    amat::Array2d<a_real>& up);

}

#endif

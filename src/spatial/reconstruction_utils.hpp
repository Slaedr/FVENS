/** \file
 * \brief Some small functions required for reconstruction
 * \author Aditya Kashi
 */

#ifndef FVENS_RECONSTRUCTION_UTILS_H
#define FVENS_RECONSTRUCTION_UTILS_H

#include <Eigen/Core>
#include <aconstants.hpp>

namespace fvens {

/// Reconstructs a face value by a "limited linear" extrapolation
/** Note that depending on the limiter, this could be nonlinear.
 */
template <typename scalar, int nvars>
static inline scalar
linearExtrapolate(const scalar ucell,                          ///< Relevant cell centred value
                  const GradBlock_t<scalar,NDIM,nvars>& grad,  ///< Gradients
                  const int ivar,                              ///< Index of physical variable to be
                  ///<  reconstructed
                  const freal lim,                            ///< Limiter value
                  const scalar *const gp,                      ///< Quadrature point coords
                  const scalar *const rc                       ///< Cell centre coords
                  )
{
	scalar uface = ucell;
	for(int idim = 0; idim < NDIM; idim++)
		uface += lim*grad(idim,ivar)*(gp[idim] - rc[idim]);
	return uface;
}

/// Returns the squared magnitude of the gradient of a variable
/** \param[in] grad The gradient array
 * \param[in] ivar The index of the physical variable whose gradient magnitude is needed
 */
template <typename scalar, int nvars>
static inline scalar gradientMagnitude2(const GradBlock_t<scalar,NDIM,nvars>& grad, const int ivar)
{
	scalar res = 0;
	for(int j = 0; j < NDIM; j++)
		res += grad(j,ivar)*grad(j,ivar);
	return res;
}

}

#endif

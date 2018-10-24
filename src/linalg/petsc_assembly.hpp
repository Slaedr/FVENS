/** \file
 * \brief Functions for assembling PETSc objects using the spatial discretization
 */

#ifndef FVENS_PETSC_ASSEMBLY
#define FVENS_PETSC_ASSEMBLY

#include <petscmat.h>
#include "spatial/aspatial.hpp"

namespace fvens {

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
 * \param[out] dtm Local time steps are stored in this
 */
template <typename scalar, int nvars>
StatusCode assemble_residual(const Spatial<scalar,nvars> *const spatial,
                             const Vec uvec, Vec __restrict rvec,
                             const bool gettimesteps, std::vector<a_real>& dtm);

template <typename scalar, int nvars>
StatusCode assemble_jacobian(const Spatial<scalar,nvars> *const spatial, const Vec uvec, Mat A);

}

#endif

/** \file
 * \brief Operations for computing viscous flux at a point
 * \author Aditya Kashi
 */

#ifndef FVENS_VISCOUSPHYSICS_H
#define FVENS_VISCOUSPHYSICS_H

#include "aphysics.hpp"

namespace fvens {

/// Computes primitive-2 variables and temperature grad from conserved variables and primitive grads
/** The output variables can then be used to compute unique face gradients using
 *   finite volume techniques such as modified averaging, for example.
 *   
 * Corresponding input and output variables can point to the same locations.
 * 
 * \param[in] physics The gas physics context to use
 * \param[in] ucl Cell-centred conserved variables on left side of the face
 * \param[in] ucr Cell-centred conserved variables on right side of the face
 * \param[in] gradl Gradients of primitive variables in left cell; must be row-major ndim x nvars
 * \param[in] gradr Gradients of primitive variables in right cell; must be row-major ndim x nvars
 * \param[in|out] uctl On output, primitive 2 variables in left cell
 * \param[in|out] uctr On output, primitive 2 variables in right cell
 * \param[in|out] gradtl On output, primitive 2 gradients in left cell (row-major)
 * \param[in|out] gradtr On output, primitive 2 gradients in right cell (row-major)
 *
 * gradl and gradr can be nullptrs if second-order accuracy is not required, in which case
 *  gradtl and gradtr are not touched.
 *
 * Note that `primtive 2' variables are density, velocities and temperature.
 */
template<typename scalar, int ndim, bool secondOrderRequested>
void getPrimitive2StatesAndGradients(const IdealGasPhysics<scalar>& physics,
                                     const scalar *const ucl, const scalar *const ucr,
                                     const scalar *const gradl, const scalar *const gradr,
                                     scalar *const uctl, scalar *const uctr,
                                     scalar *const gradtl, scalar *const gradtr);

/// Computes viscous flux across a face at one point
/** Note that the flux computed here must then be integrated on the face
 *  and *subtracted from* the residual, where residual is defined as
 * \f$ r(u) = \sum_K \int_{\partial K} F \hat{n} d\gamma \f$ (where $f K $f denotes a cell),
 * ie, it is the sum of all *outgoing* fluxes from each cell.
 *  
 * \param[in] physics The gas physics context to use
 * \param[in] n Normal vector to the face
 * \param[in] grad Unique gradients of primitive 2 variables at the face quadrature point
 * \param[in] ul Left state of faces (conserved variables)
 * \param[in] ul Right state of faces (conserved variables)
 * \param[in,out] vflux On output, contains the viscous flux across the face
 */
template<typename scalar, int ndim, int nvars, bool constVisc>
void computeViscousFlux(const IdealGasPhysics<scalar>& physics, const scalar *const n,
                        const scalar grad[ndim][nvars],
                        const scalar *const ul, const scalar *const ur,
                        scalar *const __restrict vflux);

/// Computes the Jacobians of the viscous flux w.r.t. left and right cell-centered states
/** \todo Test this routine
 * \param[in] jphysics The physics context to use
 * \param[in] n Normal vector
 * \param[in] ul Left conserved state
 * \param[in] ur Right conserved state
 * \param[in] grad The unique gradient at the face
 * \param[in] dgradl The Jacobian of the face gradient w.r.t. left conserved state
 * \param[in] dgradr The Jacobian of the face gradient w.r.t. right conserved state
 * \param[in|out] dvfi Jacobian of viscous flux across the face w.r.t. left conserved state
 * \param[in|out] dvfj Jacobian of viscous flux across the face w.r.t. right conserved state
 */
template<typename scalar, int ndim, int nvars, bool constVisc>
void computeViscousFluxJacobian(const IdealGasPhysics<scalar>& jphysics,
                                const scalar *const n,
                                const scalar *const ul, const scalar *const ur,
                                const scalar grad[ndim][nvars],
                                const scalar dgradl[ndim][nvars][nvars],
                                const scalar dgradr[ndim][nvars][nvars],
                                scalar *const __restrict dvfi, scalar *const __restrict dvfj);

}

#endif

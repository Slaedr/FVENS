/** \file mathutils.hpp
 * \brief Provides convenient functions for certain geometric or calculus operations
 * \author Aditya Kashi
 */

#ifndef FVENS_MATHUTILS_H
#define FVENS_MATHUTILS_H

#include "aconstants.hpp"

namespace fvens {

/// Fill a raw array of scalars with zeros
template <typename scalar>
inline void zeros(scalar *const __restrict a, const a_int n) {
	for(a_int i = 0; i < n; i++)
		a[i] = 0;
}

/// Returns a dot product computed between the first NDIM components of the two vectors.
template <typename scalar>
inline scalar dimDotProduct(const scalar *const u, const scalar *const v)
{
	scalar dot = 0;
	for(int i = 0; i < NDIM; i++)
		dot += u[i]*v[i];
	return dot;
}

/** Computes the Cartesian components of a vector given its magnitude and direction
 *
 * \param mag Magnitude of the vector
 * \param dir Unit vector in the direction of the vector
 * \param [out] vec Cartesian components of the vector
 */
template <typename scalar>
inline void getComponentsCartesian(const scalar mag, const scalar dir[NDIM], scalar vec[NDIM])
{
	zeros(vec,NDIM);
	const scalar cosphi = (NDIM == 3) ? sqrt(dir[0]*dir[0]+dir[1]*dir[1]) : 1.0;
	const scalar sinphi = (NDIM == 3) ? dir[2] : 0.0;
	if(NDIM == 3)
		vec[2] = mag*sinphi;
	vec[0] = mag*cosphi*dir[0];
	vec[1] = mag*cosphi*dir[1];
}

/// Returns the derivatives of f/g given the derivatives of f and g (for NVARS components)
/** \note The result is added to the output array dq; so prior contents will affect the outcome.
 * \note It is possible for the output array dq to point to the same location as
 * one of the input arrays. In that case, the entire array should overlap, NOT only a part of it.
 */
template <typename scalar>
inline void getQuotientDerivatives(const scalar f, const scalar *const df, 
                                   const scalar g, const scalar *const dg,
                                   scalar *const __restrict dq)
{
	for(int i = 0; i < NVARS; i++)
		dq[i] += (df[i]*g-f*dg[i])/(g*g);
}

}

#endif

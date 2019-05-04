/** \file
 * \brief Utilities to inter-operate with ADOL-C adoubles
 */

#ifndef FVENS_ADOLC_UTILS_H
#define FVENS_ADOLC_UTILS_H

#include "aconstants.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

/// Returns the value of the input as freal
template <typename scalar>
static inline freal getvalue(const scalar x);

// Trivial implementations

template <>
freal getvalue(const freal x) {
	return x;
}

// ADOL-C implementations (incomplete!)

#ifdef USE_ADOLC

template <>
freal getvalue(const adouble x) {
	return x.value();
}

#endif

}
#endif

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

/// Returns the value of the input as a_real
template <typename scalar>
static inline a_real getvalue(const scalar x);

// Trivial implementations

template <>
a_real getvalue(const a_real x) {
	return x;
}

// ADOL-C implementations (incomplete!)

#ifdef USE_ADOLC

template <>
a_real getvalue(const adouble x) {
	return x.value();
}

#endif

}
#endif

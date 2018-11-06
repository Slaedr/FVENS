/** \file
 * \brief Utilities to inter-operate with ADOL-C adoubles
 */

#ifndef FVENS_ADOLC_UTILS_H
#define FVENS_ADOLC_UTILS_H

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

/// Returns the value of the input as a_real
template <typename scalar>
static a_real getvalue(const scalar x);

#ifdef USE_ADOLC
template <>
a_real getvalue(const adouble x) {
	return x.value();
}
#endif

template <>
a_real getvalue(const a_real x) {
	return x;
}

}
#endif

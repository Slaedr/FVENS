/** \file adolc_eigen.hpp
 * \brief Provides machinery needed for use of ADOL-C adouble with Eigen
 *
 * Refer to https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html for details.
 */

#ifndef ADOLCSUPPORT_H
#define ADOLCSUPPORT_H

//#define ADOLC_TAPELESS

/// Since ADOL-C does not define a namespace, we do
namespace adolc {

#include <adolc/adouble.h>

#ifdef _OPENMP
#include <adolc/adolc_openmp.h>
#endif

}

#include <Eigen/Core>

namespace Eigen {

/// Scalar traits required by Eigen
template<> struct NumTraits<adolc::adouble>
	: NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef adolc::adouble Real;
	typedef adolc::adouble NonInteger;
	typedef adolc::adouble Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 3,
		MulCost = 3
	};
};

}

inline const adouble& conj(const adouble& x)  { return x; }
inline const adouble& real(const adouble& x)  { return x; }
inline adouble imag(const adouble&)    { return 0.; }
inline adouble abs(const adouble&  x)  { return fabs(x); }
inline adouble abs2(const adouble& x)  { return x*x; }

#endif // ADOLCSUPPORT_H

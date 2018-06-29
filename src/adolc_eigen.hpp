/** \file adolc_eigen.hpp
 * \brief Provides machinery needed for use of ADOL-C adouble with Eigen
 *
 * Refer to https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html for details.
 */

#ifndef ADOLCSUPPORT_H
#define ADOLCSUPPORT_H

//#define ADOLC_TAPELESS

// Unfortunately, ADOL-C does not define a namespace, so careful with names!
// Eg., fabs below must actually be fabs from ADOL-C

#include <adolc/adouble.h>

#ifdef _OPENMP
#include <adolc/adolc_openmp.h>
#endif

#include <Eigen/Core>

namespace Eigen {

/// Scalar traits required by Eigen
template<> struct NumTraits<adouble>
	: NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef adouble Real;
	typedef adouble NonInteger;
	typedef adouble Nested;
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
// Here, fabs must be from ADOL-C
inline adouble abs(const adouble&  x)  { return fabs(x); }
inline adouble abs2(const adouble& x)  { return x*x; }

#endif // ADOLCSUPPORT_H

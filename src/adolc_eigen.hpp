/** \file adolc_eigen.hpp
 * \brief Provides machinery needed for use of ADOL-C adouble with Eigen
 *
 * Refer to https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html for details.
 */

#ifndef ADOLCSUPPORT_H
#define ADOLCSUPPORT_H

//#define ADOLC_TAPELESS

/// Since ADOL-C does not define a namespace, we do
/// Use adouble etc as adolc::adouble etc in code
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

namespace fvens {
	
inline const adolc::adouble& conj(const adolc::adouble& x)  { return x; }
inline const adolc::adouble& real(const adolc::adouble& x)  { return x; }
inline adolc::adouble imag(const adolc::adouble&)    { return 0.; }
inline adolc::adouble abs(const adolc::adouble&  x)  { return adolc::fabs(x); }
inline adolc::adouble abs2(const adolc::adouble& x)  { return x*x; }

}

#endif // ADOLCSUPPORT_H

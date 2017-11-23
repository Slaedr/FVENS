/** \file aconstants.hpp
 * \brief Defines some macro constants and typedefs used throughout the code
 * \author Aditya Kashi
 */

#ifndef ACONSTANTS_H

#define ACONSTANTS_H 1

/*
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <functional>
#include <time.h>
#include <sys/time.h>

// for floating point exceptions
#ifdef DEBUG
#include <fenv.h>
#endif
*/

#define PI 3.14159265358979323846
#define SQRT3 1.73205080756887729353

/// tolerance to check if something is zero, ie, machine epsilon
#define ZERO_TOL 2.2e-16

/// A small number likely smaller than most convergence tolerances
#define A_SMALL_NUMBER 1e-12

/// Number of spatial dimensions in the problem
#define NDIM 2

/// Number of coupled variables for compressible flow computations
#define NVARS 4

/// Number of quadrature points in each face
#define NGAUSS 1

#ifndef MESHDATA_DOUBLE_PRECISION
#define MESHDATA_DOUBLE_PRECISION 20
#endif

// We don't want any built-in threading in Eigen interfering with ours
#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace acfd
{
#define DOUBLE_PRECISION 1
	
	/// The floating-point type to use for all float computations
	typedef double a_real;

	/// Integer type to use for indexing etc
	/** Using signed types for this might be better than using unsigned types,
	 * eg., to iterate backwards over an entire array (down to index 0).
	 */
	typedef int a_int;
	
	using Eigen::Dynamic;
	using Eigen::RowMajor;
	using Eigen::ColMajor;
	using Eigen::Matrix;
	using Eigen::aligned_allocator;

	/// Multi-vector type, used for storing mesh functions like the residual
	typedef Matrix<a_real, Dynamic,Dynamic,RowMajor> MVector;

	/// A fixed-size row-major array typedef \warning Has to be column major!
	/** We could have made this row major, but Eigen complains against defining
	 * row-major matrices with only one column, as required by scalar problems.
	 */
	template<int rows, int cols>
	using FArray = Eigen::Array<a_real,rows,cols,ColMajor>;

	/// Fill a raw array of reals with zeros
	inline void zeros(a_real *const a, const a_int n) {
		for(int i = 0; i < n; i++)
			a[i] = 0;
	}

	/// A data type for error codes, mostly for use with PETSc
	typedef int StatusCode;
}

#endif

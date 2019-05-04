/** \file aconstants.hpp
 * \brief Defines some macro constants and typedefs used throughout the code
 * \author Aditya Kashi
 */

#ifndef ACONSTANTS_H
#define ACONSTANTS_H 1

// for floating point exceptions
/*#ifdef DEBUG
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

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace fvens
{
#define DOUBLE_PRECISION 1

#ifndef USE_ADOLC
using std::pow;
using std::sqrt;
using std::fabs;
#endif

/// The floating-point type to use for all float computations
/** If this is changed, the appropriate PETSc library will be required. \sa FVENS_MPI_REAL
 */
typedef double freal;

/// MPI data type corresponding to fvens::freal. Make sure to change this whenever freal is changed.
#define FVENS_MPI_REAL MPI_DOUBLE

/// Integer type to use for indexing etc
/** Using signed types for this might be better than using unsigned types,
 * eg., to iterate backwards over an entire array (down to index 0).
 * \sa FVENS_MPI_INT
 */
typedef int fint;

/// MPI data type corresponding to fvens::a_int. Make sure to keep this in sync with a_int.
#define FVENS_MPI_INT MPI_INT

/// Multi-vector type, used for storing mesh functions like the residual
template <typename scalar>
using MVector = Eigen::Matrix<scalar, Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

/// An array of fixed-size Eigen matrices each with the number of space dimensions as the size
/** It is absolutely necessary to use Eigen::aligned_allocator for std::vector s of
 * fixed-size vectorizable Eigen arrays; see
 * [this](http://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html).
 */
template <typename scalar>
using DimMatrixArray = std::vector< Eigen::Matrix<scalar,NDIM,NDIM>,
                                    Eigen::aligned_allocator<Eigen::Matrix<scalar,NDIM,NDIM>> >;

/// Fixed-size Eigen array for storing things like the spatial gradient of a vector field
template <typename scalar, int ndim, int nvars>
using GradBlock_t = Eigen::Array<scalar,ndim,nvars,Eigen::ColMajor|Eigen::DontAlign>;

/// A data type for error codes, mostly for use with PETSc
typedef int StatusCode;

}

#endif

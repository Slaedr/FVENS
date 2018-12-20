/** \file
 * \brief Tests for gradient schemes
 * \author Aditya Kashi
 */

#ifndef FVENS_TESTS_GRADIENTS_H
#define FVENS_TESTS_GRADIENTS_H

#include <string>
#include "spatial/aspatial.hpp"

namespace fvens_tests {

using fvens::a_real;
using fvens::a_int;
using Eigen::Matrix;
using Eigen::RowMajor;

/// A 'spatial discretization' that does nothing but carry out tests on gradient schemes etc
class TestSpatial : public fvens::Spatial<a_real,1>
{
public:
	TestSpatial(const fvens::UMesh2dh<a_real> *const mesh);

	fvens::StatusCode compute_residual(const Vec u, Vec residual,
	                                   const bool gettimesteps, Vec dtm) const
	{ return 0; }

	void compute_local_jacobian_interior(const a_int iface,
	                                     const a_real *const ul, const a_real *const ur,
	                                     Matrix<a_real,1,1,RowMajor>& L,
	                                     Matrix<a_real,1,1,RowMajor>& U) const
	{ }

	void compute_local_jacobian_boundary(const a_int iface,
	                                     const a_real *const ul,
	                                     Matrix<a_real,1,1,RowMajor>& L) const
	{ }

	virtual void getGradients(const fvens::MVector<a_real>& u,
	                          fvens::GradBlock_t<a_real,NDIM,1> *const grads) const
	{ }

	/// Test if weighted least-squares reconstruction is '1-exact'
	int test_oneExact(const std::string reconst_type) const;

protected:
	using fvens::Spatial<a_real,1>::m;
	using fvens::Spatial<a_real,1>::rch;
	using fvens::Spatial<a_real,1>::gr;
};

}
#endif

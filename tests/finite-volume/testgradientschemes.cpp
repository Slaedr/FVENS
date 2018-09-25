/** \file
 * \brief Implementation of tests for gradient computation schemes
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <limits>
#include <iostream>
#include "testgradientschemes.hpp"
#include "spatial/agradientschemes.hpp"
#include "spatial/areconstruction.hpp"
#include "mesh/ameshutils.hpp"
#include "utilities/afactory.hpp"

namespace fvens_tests {

using namespace fvens;

static inline a_real linearfunc(const a_real *const x) {
	static_assert(NDIM <= 3, "Only upto 3 dimensions supported.");
	const a_real coeffs[] = {2.0, 0.5, -1.0};
	const a_real constcoeff = 2.5;

	a_real val = 0;
	for(int i = 0; i < NDIM; i++)
		val += coeffs[i]*x[i];
	val += constcoeff;
	return val;
}

TestSpatial::TestSpatial(const fvens::UMesh2dh<a_real> *const mesh)
	: fvens::Spatial<a_real,1>(mesh)
{ }

int TestSpatial::test_oneExact(const std::string reconst_type) const
{
	const GradientScheme<a_real,1> *const wls
		= create_const_gradientscheme<a_real,1>(reconst_type, m, rc);

	GradArray<a_real,1> grads(m->gnelem());
	MVector<a_real> u(m->gnelem(),1);
	amat::Array2d<a_real> ug(m->gnbface(),1);
	amat::Array2d<a_real> uleft(m->gnaface(),1);
	amat::Array2d<a_real> uright(m->gnaface(),1);

	// set the field as ax+by+c
	for(a_int i = 0; i < m->gnelem(); i++)
		u(i,0) = linearfunc(&rc(i));
	for(a_int i = 0; i < m->gnbface(); i++)
		ug(i,0) = linearfunc(&rc(i+m->gnelem()));

	// get gradients
	wls->compute_gradients(u, ug, grads);

	LinearUnlimitedReconstruction<a_real,1> lur(m, rc, gr);
	lur.compute_face_values(u, ug, grads, uleft, uright);

	constexpr a_real a_epsilon = std::numeric_limits<a_real>::epsilon();
	for(a_int iface = 0; iface < m->gnbface(); iface++) {
		std::cout << uleft(iface,0) <<  " " << linearfunc(&gr[iface](0,0)) << std::endl;
		assert(std::fabs(uleft(iface,0) - linearfunc(&gr[iface](0,0))) < 10*a_epsilon);
	}
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++) {
		assert(std::fabs(uleft(iface,0)-uright(iface,0)) < 10*a_epsilon);
		assert(std::fabs(uleft(iface,0)-linearfunc(&gr[iface](0,0)) < 10*a_epsilon));
	}

	delete wls;
	return 0;
}

}

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
		= create_const_gradientscheme<a_real,1>(reconst_type, m, &rc(0,0));

	std::vector<GradBlock_t<a_real,NDIM,1>> grads(m->gnelem());
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
	wls->compute_gradients(u, ug, &grads[0]);

	LinearUnlimitedReconstruction<a_real,1> lur(m, &rc(0,0), gr);
	lur.compute_face_values(u, ug, &grads[0], uleft, uright);

	constexpr a_real a_epsilon = std::numeric_limits<a_real>::epsilon();
	a_real errnorm = 0;
	a_real lrerrnorm = 0;

	// Compute RMS errors
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++) {
		errnorm += std::pow(uleft(iface,0) - linearfunc(&gr(iface,0)),2);
	}
	for(a_int iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++) {
		lrerrnorm += std::pow(uleft(iface,0)-uright(iface,0),2);
		errnorm += std::pow(uleft(iface,0)-linearfunc(&gr(iface,0)),2);
	}
	for(a_int iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++) {
		errnorm += std::pow(uleft(iface,0) - linearfunc(&gr(iface,0)),2);
	}

	errnorm = std::sqrt(errnorm/m->gnaface());
	lrerrnorm = std::sqrt(lrerrnorm/m->gnaface());

	std::cout << "Error norm = " << errnorm << std::endl;
	std::cout << "LR error norm = " << lrerrnorm << std::endl;

	assert(errnorm < 10*a_epsilon);
	assert(lrerrnorm < 10*a_epsilon);

	delete wls;
	return 0;
}

}

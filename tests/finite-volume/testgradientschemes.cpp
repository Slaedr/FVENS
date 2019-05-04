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

static inline freal linearfunc(const freal *const x) {
	static_assert(NDIM <= 3, "Only upto 3 dimensions supported.");
	const freal coeffs[] = {2.0, 0.5, -1.0};
	const freal constcoeff = 2.5;

	freal val = 0;
	for(int i = 0; i < NDIM; i++)
		val += coeffs[i]*x[i];
	val += constcoeff;
	return val;
}

TestSpatial::TestSpatial(const fvens::UMesh<freal,NDIM> *const mesh)
	: fvens::Spatial<freal,1>(mesh)
{ }

int TestSpatial::test_oneExact(const std::string reconst_type) const
{
	const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);
	const GradientScheme<freal,1> *const wls
		= create_const_gradientscheme<freal,1>(reconst_type, m, &rc(0,0), &rcbp(0,0));

	std::vector<GradBlock_t<freal,NDIM,1>> grads(m->gnelem());
	MVector<freal> u(m->gnelem(),1);
	amat::Array2d<freal> ug(m->gnbface(),1);
	amat::Array2d<freal> uleft(m->gnaface(),1);
	amat::Array2d<freal> uright(m->gnaface(),1);

	// set the field as ax+by+c
	for(fint i = 0; i < m->gnelem(); i++)
		u(i,0) = linearfunc(&rc(i,0));
	for(fint i = 0; i < m->gnbface(); i++)
		ug(i,0) = linearfunc(&rcbp(i));

	// get gradients
	wls->compute_gradients(amat::Array2dView<freal>(&u(0,0),m->gnelem()+m->gnConnFace(),1),
	                       amat::Array2dView<freal>(&ug(0,0),m->gnbface(),1), &grads[0](0,0));

	LinearUnlimitedReconstruction<freal,1> lur(m, &rc(0,0), &rcbp(0,0), gr);
	lur.compute_face_values(u, amat::Array2dView<freal>(&ug(0,0),m->gnbface(),1), &grads[0](0,0),
	                        amat::Array2dMutableView<freal>(&uleft(0,0),m->gnaface(),1),
	                        amat::Array2dMutableView<freal>(&uright(0,0),m->gnaface(),1));

	constexpr freal a_epsilon = std::numeric_limits<freal>::epsilon();
	freal errnorm = 0;
	freal lrerrnorm = 0;

	// Compute RMS errors
	for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++) {
		errnorm += std::pow(uleft(iface,0) - linearfunc(&gr(iface,0)),2);
	}
	for(fint iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++) {
		lrerrnorm += std::pow(uleft(iface,0)-uright(iface,0),2);
		errnorm += std::pow(uleft(iface,0)-linearfunc(&gr(iface,0)),2);
	}
	for(fint iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++) {
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

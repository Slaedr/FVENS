/** \file testwallbcs.cpp
 * \brief Implements unit tests for wall boundary conditions of Euler or Navier-Stokes equations
 * \author Aditya Kashi
 * \date 2017-10
 */
#include <iostream>
#include "testwallbcs.hpp"

#define FLUX_TOL 10*ZERO_TOL

namespace fvens {
namespace fvens_tests {

int TestFlowFV::testWalls(const freal *const u) const
{
	int ierr = 0;

	for(int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		freal ug[NVARS];
		freal n[NDIM];
		for(int i = 0; i < NDIM; i++)
			n[i] = m->gfacemetric(iface,i);

		compute_boundary_state(iface, u, ug);

		freal flux[NVARS];
		inviflux->get_flux(u,ug,n,flux);
		
		if(bcs.at(m->gbtags(iface,0))->bctype == ADIABATIC_WALL_BC)
		{
			if(std::fabs(flux[0]) > FLUX_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at adiabatic wall is nonzero " << flux[0] << "\n";
			}

			if(std::fabs(flux[NVARS-1]) > FLUX_TOL) {
				ierr = 1;
				std::cerr << "! Normal energy flux at adiabatic wall is nonzero! "
				          << flux[NVARS-1] << "\n";
			}
		}

		/*if(m->gbtags(iface,0) == pconfig.isothermalwall_id)
			if(std::fabs(flux[0]) > ZERO_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at isothermal wall is nonzero!\n";
			}*/

		/*if(m->gbtags(iface,0) == pconfig.isothermalbaricwall_id)
			if(std::fabs(flux[0]) > ZERO_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at isothermalbaric wall is nonzero!\n";
			}*/
		
		if(bcs.at(m->gbtags(iface,0))->bctype == SLIP_WALL_BC)
		{
			if(std::fabs(flux[0]) > FLUX_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at slip wall is nonzero!\n";
			}

			if(std::fabs(flux[NVARS-1]) > 10*FLUX_TOL) {
				ierr = 1;
				std::cerr << "! Normal energy flux at slip wall is nonzero!\n";
			}
		}
	}

	return ierr;
}

std::array<freal,NVARS> get_test_state()
{
	const freal p_nondim = 10.0;
	std::array<freal,NVARS> u {1.0, 0.5, 0.5, p_nondim/(1.4-1.0) + 0.5*0.5 };
	return u;
}

}
}

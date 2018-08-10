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

int TestFlowFV::testWalls(const a_real *const u) const
{
	int ierr = 0;

	for(int iface = 0; iface < m->gnbface(); iface++)
	{
		a_real ug[NVARS];
		a_real n[NDIM];
		for(int i = 0; i < NDIM; i++)
			n[i] = m->gfacemetric(iface,i);

		compute_boundary_state(iface, u, ug);

		a_real flux[NVARS];
		inviflux->get_flux(u,ug,n,flux);
		
		if(bcs.at(m->gintfacbtags(iface,0))->bctype == ADIABATIC_WALL_BC)
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

		/*if(m->gintfacbtags(iface,0) == pconfig.isothermalwall_id)
			if(std::fabs(flux[0]) > ZERO_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at isothermal wall is nonzero!\n";
			}*/

		/*if(m->gintfacbtags(iface,0) == pconfig.isothermalbaricwall_id)
			if(std::fabs(flux[0]) > ZERO_TOL) {
				ierr = 1;
				std::cerr << "! Normal mass flux at isothermalbaric wall is nonzero!\n";
			}*/
		
		if(bcs.at(m->gintfacbtags(iface,0))->bctype == SLIP_WALL_BC)
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

std::array<a_real,NVARS> get_test_state()
{
	const a_real p_nondim = 10.0;
	std::array<a_real,NVARS> u {1.0, 0.5, 0.5, p_nondim/(1.4-1.0) + 0.5*0.5 };
	return u;
}

}
}

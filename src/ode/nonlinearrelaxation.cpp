/** \file
 * \brief Implementation of some nonlinear relaxation schemes
 * \author Aditya Kashi
 * \date 2018-11
 */

#include <iostream>
#include "nonlinearrelaxation.hpp"
#include "physics/aphysics_defs.hpp"

namespace fvens {

template <int nvars>
NonlinearUpdate<nvars>::~NonlinearUpdate() { }

template <int nvars>
FlowSimpleUpdate<nvars>::FlowSimpleUpdate(const IdealGasPhysics<a_real> p, const a_real mf)
	: phy{p}, minfactor{mf}
{
	if(minfactor <= 0 || minfactor > 1.0)
		throw std::domain_error("Minimum relaxation factor is invalid!");
}

template <int nvars>
a_real FlowSimpleUpdate<nvars>::getLocalRelaxationFactor(const a_real du[nvars],
                                                         const a_real u[nvars]) const
{
	const a_real p = phy.getPressureFromConserved(u);
	const a_real dp = std::abs(phy.getDeltaPressureFromConserved(u, du)) / p;
	const a_real drho = std::abs(du[0])/u[0];
	const a_real danger_level = std::max(dp,drho);

	a_real omega = minfactor;
	if(danger_level < 1.0-minfactor)
		omega = 1.0-danger_level;

	return omega;
}

template class NonlinearUpdate<1>;
template class NonlinearUpdate<NDIM+2>;
template class FullUpdate<1>;
template class FullUpdate<NDIM+2>;
template class FlowSimpleUpdate<1>;
template class FlowSimpleUpdate<NDIM+2>;

}

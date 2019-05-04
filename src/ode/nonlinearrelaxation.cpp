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
FlowSimpleUpdate<nvars>::FlowSimpleUpdate(const IdealGasPhysics<freal> p, const freal mf)
	: phy{p}, minfactor{mf}
{
	if(minfactor <= 0 || minfactor > 1.0)
		throw std::domain_error("Minimum relaxation factor is invalid!");
}

template <int nvars>
freal FlowSimpleUpdate<nvars>::getLocalRelaxationFactor(const freal du[nvars],
                                                         const freal u[nvars]) const
{
	const freal p = phy.getPressureFromConserved(u);
	const freal dp = std::abs(phy.getDeltaPressureFromConserved(u, du)) / p;
	const freal drho = std::abs(du[0])/u[0];
	const freal danger_level = std::max(dp,drho);

	freal omega = minfactor;
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

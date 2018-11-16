/** \file
 * \brief Implementation of some nonlinear relaxation schemes
 * \author Aditya Kashi
 * \date 2018-11
 */

#include <iostream>
#include "nonlinearrelaxation.hpp"
#include "physics/aphysics_defs.hpp"

namespace fvens {

template <int ndim>
FlowSimpleRelaxation<ndim>::FlowSimpleRelaxation(const IdealGasPhysics<a_real> p, const a_real mf)
	: phy{p}, minfactor{mf}
{ }

template <int ndim>
a_real FlowSimpleRelaxation<ndim>::getLocalRelaxationFactor(const a_real dur[nvars],
                                                            const a_real ur[nvars]) const
{
	// using Eigen::Array;
	// using Eigen::Map;
	// Map<const Array<a_real,nvars,1>> du(dur);
	// Map<const Array<a_real,nvars,1>> uold(ur);
	// const Array<a_real,nvars,1> unew = uold+du;
	const a_real p = phy.getPressureFromConserved(ur);
	const a_real dp = phy.getDeltaPressureFromConserved(ur, dur);
	return dp;
}

}

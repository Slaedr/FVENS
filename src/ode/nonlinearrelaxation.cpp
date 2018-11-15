/** \file
 * \brief Implementation of some nonlinear relaxation schemes
 * \author Aditya Kashi
 * \date 2018-11
 */

#include "nonlinearrelaxation.hpp"

namespace fvens {

FlowSimpleRelaxation::FlowSimpleRelaxation(const a_real mf)
	: minfactor{mf}
{ }

FlowSimpleRelaxation::getLocalRelaxationFactor(const a_real du[nvars], const a_real u[nvars])
{
}

}

/** \file
 * \brief Methods to underrelax the nonlinear update for better stability or convergence.
 *
 * In this file, by nonlinear relaxation we mean the relaxation factor used for scaling a computed
 * update before adding to the current iterate of the approximate solution at a nonlinear solver step.
 * 
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_NONLINEAR_RELAXATION_H
#define FVENS_NONLINEAR_RELAXATION_H

#include "aconstants.hpp"

namespace fvens {

/// Abstract base class for computation of a (local) relaxation factor given a state and an update
template <int nvars>
class NonlinearRelaxation
{
public:
	virtual ~NonlinearRelaxation();

	/// Computes a local relaxation factor based on the local state and the local update
	virtual a_real getLocalRelaxationFactor(const a_real du[nvars], const a_real u[nvars]) const = 0;
};

class FlowSimpleRelaxation : public NonlinearRelaxation<NVARS>
{
public:
	FlowSimpleRelaxation(const a_real min_factor);

	a_real getLocalRelaxationFactor(const a_real du[nvars], const a_real u[nvars]) const;

protected:
	/// The minimum allowed relaxation factor
	const a_real minfactor;
};

}

#endif

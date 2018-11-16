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
#include "physics/aphysics.hpp"

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

/// A simple under-relaxation scheme for implicit flow solves which attempts prevent large changes
///  in density and pressure
/** The template parameter is the number of spatial dimensions in the problem.
 */
template <int ndim>
class FlowSimpleRelaxation : public NonlinearRelaxation<ndim+2>
{
public:
	static constexpr int nvars = ndim+2;

	FlowSimpleRelaxation(const IdealGasPhysics<a_real> physics, const a_real min_factor);

	a_real getLocalRelaxationFactor(const a_real du[nvars], const a_real u[nvars]) const;

protected:
	/// Gas physics context
	const IdealGasPhysics<a_real> phy;
	/// The minimum allowed relaxation factor
	const a_real minfactor;
};

}

#endif

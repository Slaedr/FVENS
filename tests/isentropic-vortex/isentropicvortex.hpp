/** \file isentropicvortex.hpp
 * \brief Definitions for isentropic vortex test
 * \author Aditya Kashi
 * \date 2018-05
 */

#ifndef FVENS_ISENTROPIC_VORTEX_H
#define FVENS_ISENTROPIC_VORTEX_H

#include <array>
#include "mesh/amesh2dh.hpp"

namespace fvens{
namespace fvens_tests {

/// Physical configuration for the isentropic vortex problem
struct IsenVortexConfig {
	a_real gamma;                ///< Adiabatic index
	a_real Minf;                 ///< Free-stream Mach number
	std::array<double,2> centre; ///< Initial position of vortex centre
	a_real strength;             ///< Strength of the vortex
	a_real clength;              ///< Characteristic length in the domain
	a_real sigma;                ///< Std deviation of the vortex
	a_real aoa;                  ///< Angle of attack IN RADIANS
};

/// Specification of isentropic vortex problems solution
/** Ref: \ref survey_isentropicvortex (Spiegel, Huynh and DeBonis, AIAA)
 */
class IsentropicVortexProblem
{
public:
	IsentropicVortexProblem(const IsenVortexConfig config);

	/// Computes initial condition and exact solution for a given mesh and final time
	void getInitialConditionAndExactSolution(const UMesh2dh<a_real>& mesh, const a_real time,
	                                         a_real *const u, a_real *const uexact) const;

protected:
	/// Parameters defining the problem
	IsenVortexConfig conf;

	/// Free-stream pressure
	a_real pinf;

	/// Computes vortex intensity at a given location
	a_real computeOmega(const std::array<a_real,2> r) const;

	/// Computes velocity
	/** \param r Position
	 * \param omega Vortex intensity \sa computeOmega
	 */
	std::array<a_real,2> computeVelocity(const std::array<a_real,2> r, const a_real omega) const;

	/// Computes the flow state as non-dimensional conserved variables at a given location
	void computeFlowState(const std::array<a_real,2> cellcentre,
	                      a_real *const u) const;
};

}
}
#endif

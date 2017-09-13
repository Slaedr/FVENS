/** \file aphysics.hpp
 * \brief Provides analytical flux computation contexts
 * \author Aditya Kashi
 * \date 2017 May 12
 */

#ifndef __APHYSICS_H
#define __APHYSICS_H

#include "aconstants.hpp"

namespace acfd {

/// Abstract class providing analytical fluxes and their Jacobians etc
class Physics
{
public:
	/// Computes the flux vector along some direction
	virtual void evaluate_normal_flux(const a_real *const u, const a_real* const n, 
			a_real *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void evaluate_normal_jacobian(const a_real *const u, const a_real* const n, 
			a_real *const dfdu) const = 0;
};
	
/// Flow-physics-related computation for single-phase ideal gas
class IdealGasPhysics : public Physics
{
protected:
	/// Adiabatic constant
	const a_real g;

public:
	IdealGasPhysics(a_real _g) : g(_g)
	{ }

	a_real gamma() const { return g; }

	/// Computes the analytical convective flux across a face oriented in some direction
	void evaluate_normal_flux(const a_real *const u, const a_real* const n, 
			a_real *const __restrict flux) const;
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void evaluate_normal_jacobian(const a_real *const u, const a_real* const n, 
			a_real *const __restrict dfdu) const;

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/*void convertConservedToPrimitive(const a_real *const uc, a_real *const up);

	/// Convert primitive variables to conserved
	void convertPrimitiveToConserved(const a_real *const up, a_real *const uc);

	/// Compute temperature from conserved variables
	a_real getTemperatureFromConserved(const a_real *const uc);

	/// Compute temperature from primitive variables
	a_real getTemperatureFromPrimitive(const a_real *const up);

	/// Compute temperature spatial derivative from conserved variables and their spatial derivatives
	a_real getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
			const a_real *const guc);

	/// Compute spatial derivative of from conserved variables and
	/// spatial derivatives of *primitive* variables
	a_real getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
			const a_real *const gup);

	/// Computes dynamic viscosity using Sutherland's law
	a_real getViscosityFromConserved(const a_real *const uc);*/
};

}
#endif

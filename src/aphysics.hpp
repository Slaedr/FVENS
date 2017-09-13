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
/** \warning The non-dimensionalization assumed is from free-stream velocities and temperatues,
 * as given in section 4.14.2 of \cite matatsuka
 */
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
	void convertConservedToPrimitive(const a_real *const uc, a_real *const up)
	{
		up[0] = uc[0];
		a_real rhovmag2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++) {
			rhovmag2 += uc[idim]*uc[idim];
			up[idim] = uc[idim]/uc[0];
		}
		up[NDIM+1] = (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
	}

	/// Convert primitive variables to conserved
	void convertPrimitiveToConserved(const a_real *const up, a_real *const uc)
	{
		uc[0] = up[0];
		a_real vmag2 = 0;
		for(int idim = 1; idim < NDIM; idim++) {
			vmag2 += up[idim]*up[idim];
			uc[idim] = up[0]*up[idim];
		}
		uc[NDIM+1] = up[NDIM+1]/(g-1.0) + 0.5*up[0]*vmag2;
	}

	/// Computes non-dimensional temperature from non-dimensional conserved variables
	/** \sa IdealGasPhysics
	 */
	a_real getTemperatureFromConserved(const a_real *const uc, const a_real Minf)
	{
		a_real rhovmag2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++)
			rhovmag2 += uc[idim]*uc[idim];
		a_real p = (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
		return g*Minf*Minf*p/uc[0];
	}

	/// Computes non-dimensional temperature from non-dimensional primitive variables
	a_real getTemperatureFromPrimitive(const a_real *const up, const a_real Minf)
	{
		return g*Minf*Minf*up[NDIM+1]/up[0];
	}

	/// Compute temperature spatial derivative from conserved variables and their spatial derivatives
	a_real getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
			const a_real *const guc);

	/// Compute spatial derivative of from conserved variables and
	/// spatial derivatives of *primitive* variables
	a_real getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
			const a_real *const gup);

	/// Computes dynamic viscosity using Sutherland's law
	a_real getViscosityFromConserved(const a_real *const uc);
};

}
#endif

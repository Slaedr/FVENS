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
/** The non-dimensionalization assumed is from free-stream velocities and temperatues,
 * as given in section 4.14.2 of \cite matatsuka.
 * Note that several computations here depend on the exact non-dimensionalization scheme used.
 */
class IdealGasPhysics : public Physics
{
public:
	IdealGasPhysics(const a_real _g, const a_real M_inf, 
			const a_real T_inf, const a_real Re_inf, const a_real _Pr) 
		: g{_g}, Minf{M_inf}, Tinf{T_inf}, Reinf{Re_inf}, Pr{_Pr}, sC{110.5}
	{
		std::cout << " IdealGasPhysics: Physical parameters:\n";
		std::cout << "  Adiabatic index = " <<g << ", M_infty = " <<Minf << ", T_infty = " << Tinf
			<< "\n   Re_infty = " << Reinf << ", Pr = " << Pr << std::endl;
	}

	/// Computes the analytical convective flux across a face oriented in some direction
	void evaluate_normal_flux(const a_real *const u, const a_real* const n, 
			a_real *const __restrict flux) const;
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void evaluate_normal_jacobian(const a_real *const u, const a_real* const n, 
			a_real *const __restrict dfdu) const;

	a_real getPressureFromConserved(const a_real *const uc) const
	{
		a_real rhovmag2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++)
			rhovmag2 += uc[idim]*uc[idim];
		return (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
	}

	a_real getSoundSpeedFromConserved(const a_real *const uc) const
	{
		return std::sqrt(g * getPressureFromConserved(uc)/uc[0]);
	}

	a_real getEntropyFromConserved(const a_real *const uc) const
	{
		return getPressureFromConserved(uc)/std::pow(uc[0],g);
	}

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void convertConservedToPrimitive(const a_real *const uc, a_real *const up) const
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
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void convertPrimitiveToConserved(const a_real *const up, a_real *const uc) const
	{
		uc[0] = up[0];
		a_real vmag2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++) {
			vmag2 += up[idim]*up[idim];
			uc[idim] = up[0]*up[idim];
		}
		uc[NDIM+1] = up[NDIM+1]/(g-1.0) + 0.5*up[0]*vmag2;
	}

	/// Computes density from pressure and temperature using ideal gas relation;
	/// All quantities are non-dimensional
	a_real getDensityFromPressureTemperature(const a_real pressure, const a_real temperature) const
	{
		return g*Minf*Minf*pressure/temperature;
	}

	/// Computes non-dimensional temperature from non-dimensional conserved variables
	/** \sa IdealGasPhysics
	 */
	a_real getTemperatureFromConserved(const a_real *const uc) const
	{
		const a_real p = getPressureFromConserved(uc);
		return g*Minf*Minf*p/uc[0];
	}

	/// Computes non-dimensional temperature from non-dimensional primitive variables
	a_real getTemperatureFromPrimitive(const a_real *const up) const
	{
		return g*Minf*Minf*up[NDIM+1]/up[0];
	}

	/// Compute non-dim temperature spatial derivative 
	/// from non-dim conserved variables and their spatial derivatives
	a_real getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
			const a_real *const guc) const
	{
		const a_real p = getPressureFromConserved(uc);
		a_real term1 = 0, term2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++)
		{
			term1 += uc[idim]*guc[idim];
			term2 += uc[idim]*uc[idim];
		}
		a_real dpdx = (g-1.0) * (guc[NDIM-1] - 0.5/(uc[0]*uc[0])*(2*uc[0]*term1 - term2*guc[0]));
		return (uc[0]*dpdx - p*guc[0])/(uc[0]*uc[0]) * g*Minf*Minf;
	}

	/// Compute spatial derivative of non-dim temperature from non-dim conserved variables and
	/// spatial derivatives of non-dim *primitive* variables
	a_real getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
			const a_real *const gup) const
	{
		const a_real p = getPressureFromConserved(uc);
		return (uc[0]*gup[NDIM+1] - p*gup[0]) / (uc[0]*uc[0]) * g*Minf*Minf;
	}

	/// Computes total energy from a vector of density, velocities and temperature
	/** All quantities are non-dimensional.
	 */
	a_real getEnergyFromPrimitive2(const a_real *const upt) const
	{
		const a_real p = upt[0]*upt[NDIM+1]/(g*Minf*Minf);
		a_real vmag2 = 0;
		for(int idim = 1; idim < NDIM+1; idim++)
			vmag2 += upt[idim]*upt[idim];
		return p/(g-1.0) + 0.5*upt[0]*vmag2;
	}

	/// Computes non-dimensional viscosity using Sutherland's law from conserved variables
	a_real getViscosityFromConserved(const a_real *const uc) const
	{
		a_real T = getTemperatureFromConserved(uc);
		return (1+sC/Tinf)/(T+sC/Tinf) * std::pow(T,1.5) / Reinf;
	}

	a_real getThermalDiffusivityFromConserved(const a_real *const uc) const
	{
		a_real muhat = getViscosityFromConserved(uc);
		return g*muhat / ((g-1.0)*Pr);
	}

	/// Adiabatic constant
	const a_real g;
	/// Free-stream Mach number
	const a_real Minf;
	/// Free-stream static temperature
	const a_real Tinf;
	/// Free-stream Reynolds number
	const a_real Reinf;
	/// Prandtl number
	const a_real Pr;
	/// Sutherland constant
	const a_real sC;
};

}
#endif

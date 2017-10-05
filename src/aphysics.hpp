/** \file aphysics.hpp
 * \brief Provides analytical flux computation contexts
 *
 * Some functions have been defined in this file itself for inlining;
 * others have been defined in the [implementation file](\ref aphysics.cpp).
 *
 * \author Aditya Kashi
 * \date 2017 May 12
 */

#ifndef APHYSICS_H
#define APHYSICS_H

#include "aconstants.hpp"

namespace acfd {

/// Abstract class providing analytical fluxes and their Jacobians etc
class Physics
{
public:
	/// Computes the flux vector along some direction
	virtual void getNormalFluxFromConserved(const a_real *const u, const a_real* const n, 
			a_real *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void getJacobianNormalFluxWrtConserved(const a_real *const u, const a_real* const n, 
			a_real *const dfdu) const = 0;
};
	
/// Flow-physics-related computation for single-phase ideal gas
/** The non-dimensionalization assumed is from free-stream velocities and temperatues,
 * as given in section 4.14.2 of \cite matatsuka.
 * Note that several computations here depend on the exact non-dimensionalization scheme used.
 *
 * "Primitive-2" variables are density, velocities and temperature, as opposed to
 * "primitive" variables which are density, velocities and pressure.
 */
class IdealGasPhysics : public Physics
{
public:
	IdealGasPhysics(const a_real _g, const a_real M_inf, 
			const a_real T_inf, const a_real Re_inf, const a_real _Pr);

	/// Computes normal flux efficiently using specific data
	/** \param[in] uc Vector of conserved variables
	 * \param[in] n Normal vector
	 * \param[in] vn Normal velocity (w.r.t. the the normal n above)
	 * \param[in] p Pressure
	 * \param[in|out] flux Output vector for the flux; note that any pre-existing contents
	 *   will be replaced!
	 */
	void getNormalFluxEfficiently(const a_real *const uc, const a_real *const n,
			const a_real vn, const a_real p, a_real *const __restrict flux) const
		__attribute__((always_inline));

	/// Computes the analytical convective flux across a face oriented in some direction
	void getNormalFluxFromConserved(const a_real *const u, const a_real* const n, 
			a_real *const __restrict flux) const;
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void getJacobianNormalFluxWrtConserved(const a_real *const u, const a_real* const n, 
			a_real *const __restrict dfdu) const;

	/// Outputs various quantities, especially needed by numerical fluxes
	/** \param[in] uc Conserved variables
	 * \param[in] n Normal vector
	 * \param[out] vx X-velocity
	 * \param[out] vy Y-velocity
	 * \param[out] vn Normal velocity
	 * \param[out] vm2 Square of velocity magnitude
	 * \param[out] p Pressure
	 * \param[out] H Specific enthalpy
	 */
	void getVarsFromConserved(const a_real *const uc, const a_real *const n,
			a_real& vx, a_real& vy,
			a_real& vn,
			a_real& p, a_real& H ) const
		__attribute__((always_inline));

	/// Computes derivatives of variables computed in getVarsFromConserved
	/** \warning The result is added to the original content of the  output variables!
	 */
	void getJacobianVarsWrtConserved(const a_real *const uc, const a_real *const n,
		a_real dvx[NVARS], a_real dvy[NVARS], a_real dvn[NVARS],
		a_real dp[NVARS], a_real dH[NVARS]) const;

	/// Computes pressure from conserved variables
	a_real getPressureFromConserved(const a_real *const uc) const;
	
	/// Derivative of pressure w.r.t. conserved variables
	/** Note that the derivative is added to the second argument - the latter is not zeroed
	 * or anything.
	 */
	void getJacobianPressureWrtConserved(const a_real *const uc, a_real *const __restrict dp) const;

	/// Computes speed of sound from conserved variables
	a_real getSoundSpeedFromConserved(const a_real *const uc) const;

	/// Derivative of sound speed w.r.t. conserved variables
	void getJacobianSoundSpeedWrtConserved(const a_real *const uc,
			a_real *const __restrict dc) const;

	/// Computes speed of sound from conserved variables and pressure
	a_real getSoundSpeedFromConservedEfficiently(const a_real *const uc, const a_real p) const;

	/// Derivative of sound speed w.r.t. conserved variables
	void getJacobianSoundSpeedWrtConservedEfficiently(const a_real *const uc,
			const a_real p, const a_real dp[NVARS], const a_real c,
			a_real *const __restrict dc) const;

	/// Computes an entropy \f$ p/ \rho^\gamma \f$ from conserved variables
	a_real getEntropyFromConserved(const a_real *const uc) const;

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void convertConservedToPrimitive(const a_real *const uc, a_real *const up) const;
	
	/// Convert conserved variables to primitive-2 variables; depends on non-dimensionalization
	void convertConservedToPrimitive2(const a_real *const uc, a_real *const up) const;
	
	/// Computes the Jacobian matrix of the conserved-to-primitive-2 transformation
	/** \f$ \partial \mathbf{u}_{prim2} / \partial \mathbf{u}_{cons} \f$. 
	 * The output is stored as 1D rowmajor.
	 *
	 * \warning The Jacobian is *added* to the output jac. If it has garbage at the outset,
	 * it will contain garbage at the end.
	 */
	void getJacobianPrimitive2WrtConserved(const a_real *const uc, 
			a_real *const __restrict jac) const;

	/// Convert primitive variables to conserved
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void convertPrimitiveToConserved(const a_real *const up, a_real *const uc) const;

	/// Converts primitive variables to `primitive2', ie, replaces pressure with temperature
	void convertPrimitiveToPrimitive2(const a_real *const up, a_real *const up2) const;

	/// Computes density from pressure and temperature using ideal gas relation;
	/// All quantities are non-dimensional
	a_real getDensityFromPressureTemperature(const a_real pressure, const a_real temperature) const;

	/// Computes non-dimensional temperature from non-dimensional conserved variables
	/** \sa IdealGasPhysics
	 */
	a_real getTemperatureFromConserved(const a_real *const uc) const;
	
	/// Computes derivatives of temperature w.r.t. conserved variables
	/** \param[in|out] dT The derivatives are added to dT, which is not zeroed initially.
	 */
	void getJacobianTemperatureWrtConserved(const a_real *const uc, 
			a_real *const __restrict dT) const;

	/// Computes non-dimensional temperature from non-dimensional primitive variables
	a_real getTemperatureFromPrimitive(const a_real *const up) const;

	/// Compute non-dim temperature spatial derivative 
	/// from non-dim conserved variables and their spatial derivatives
	a_real getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
			const a_real *const guc) const;

	/// Compute spatial derivative of non-dim temperature from non-dim conserved variables and
	/// spatial derivatives of non-dim *primitive* variables
	a_real getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
			const a_real *const gup) const;
	
	/// Computes temerature gradients from primitive and their gradients
	a_real getGradTemperatureFromPrimitiveAndGradPrimitive(const a_real *const up,
			const a_real *const gup) const;

	/// Get primitive-2 gradients from conserved variables and their gradients
	void getGradPrimitive2FromConservedAndGradConserved(const a_real *const __restrict uc,
			const a_real *const guc, a_real *const gp) const;

	/// Computes total energy from a vector of density, velocities and temperature
	/** All quantities are non-dimensional.
	 */
	a_real getEnergyFromPrimitive2(const a_real *const upt) const;

	/// Computes non-dimensional viscosity coeff using Sutherland's law from conserved variables
	/** This is the dynamic viscosity divided by the Reynolds number
	 * when non-dimensionalized as stated in IdealGasPhysics.
	 * Note that divergence terms must still be multiplied further by -2/3 
	 * and diagonal stress terms by 2.
	 */
	a_real getViscosityCoeffFromConserved(const a_real *const uc) const;

	/// Computes derivatives of Sutherland's dynamic viscosity coeff w.r.t. conserved variables
	void getJacobianSutherlandViscosityWrtConserved(const a_real *const uc, 
			a_real *const __restrict dmu) const;
	
	/// Computes non-dimensional viscosity coeff using Sutherland's law from primitive-2 variables
	/** By viscosity coefficient, we mean dynamic viscosity divided by 
	 * the free-stream Reynolds number.
	 */
	a_real getViscosityCoeffFromPrimitive2(const a_real *const up) const;

	/// Returns non-dimensional free-stream viscosity coefficient
	a_real getConstantViscosityCoeff() const;

	/// Computes non-dimensional conductivity from non-dimensional dynamic viscosity coeff
	a_real getThermalConductivityFromViscosity(const a_real muhat) const;

	/** \brief Computes derivatives of non-dim thermal conductivity w.r.t. conserved variables
	 * given derivatives of the non-dim viscosity coeff w.r.t. conserved variables.
	 */
	void getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
			const a_real *const dmuhat, a_real *const __restrict dkhat) const;

	/// Returns the non-dimensionalized free-stream pressure
	a_real getFreestreamPressure() const;

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

inline
IdealGasPhysics::IdealGasPhysics(const a_real _g, const a_real M_inf, 
		const a_real T_inf, const a_real Re_inf, const a_real _Pr) 
	: g{_g}, Minf{M_inf}, Tinf{T_inf}, Reinf{Re_inf}, Pr{_Pr}, sC{110.5}
{
	std::cout << " IdealGasPhysics: Physical parameters:\n";
	std::cout << "  Adiabatic index = " <<g << ", M_infty = " <<Minf << ", T_infty = " << Tinf
		<< "\n   Re_infty = " << Reinf << ", Pr = " << Pr << std::endl;
}

inline
void IdealGasPhysics::getNormalFluxEfficiently(const a_real *const uc, const a_real *const n,
		const a_real vn, const a_real p, a_real *const __restrict flux) const
{
	flux[0] = vn*uc[0];
	flux[1] = vn*uc[1] + p*n[0];
	flux[2] = vn*uc[2] + p*n[1];
	flux[3] = vn*(uc[3] + p);
}

inline
void IdealGasPhysics::getVarsFromConserved(const a_real *const uc, const a_real *const n,
		a_real& vx, a_real& vy,
		a_real& vn,
		a_real& p, a_real& H ) const
{
	vx = uc[1]/uc[0]; 
	vy = uc[2]/uc[0];
	vn = vx*n[0] + vy*n[1];
	p = (g-1.0)*(uc[3] - 0.5*uc[0]*(vx*vx+vy*vy));
	H = (uc[3]+p)/uc[0];
}

inline
a_real IdealGasPhysics::getPressureFromConserved(const a_real *const uc) const
{
	a_real rhovmag2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++)
		rhovmag2 += uc[idim]*uc[idim];
	return (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
}

inline
void IdealGasPhysics::getJacobianPressureWrtConserved(const a_real *const uc, 
		a_real *const __restrict dp) const
{
	a_real rhovmag2 = 0;
	a_real drvmg2[NVARS];
	for(int i = 0; i < NVARS; i++)
		drvmg2[i] = 0;

	for(int idim = 1; idim < NDIM+1; idim++) {
		rhovmag2 += uc[idim]*uc[idim];
		drvmg2[idim] += 2.0*uc[idim];
	}
	
	dp[0] += (g-1.0)*(-0.5)* (drvmg2[0]*uc[0] - rhovmag2)/(uc[0]*uc[0]);
	dp[1] += (g-1.0)*(-0.5)* drvmg2[1]/uc[0];
	dp[2] += (g-1.0)*(-0.5)* drvmg2[2]/uc[0];
	dp[3] += (g-1.0);
}

inline
a_real IdealGasPhysics::getSoundSpeedFromConserved(const a_real *const uc) const
{
	return std::sqrt(g * getPressureFromConserved(uc)/uc[0]);
}

inline
void IdealGasPhysics::getJacobianSoundSpeedWrtConserved(const a_real *const uc,
		a_real *const __restrict dc) const
{
	a_real p =getPressureFromConserved(uc);
	a_real dp[NVARS]; for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc, dp);

	const a_real c = std::sqrt(g*p/uc[0]);
	
	dc[0] += 0.5/c * g* (dp[0]*uc[0]-p)/(uc[0]*uc[0]);
	for(int i = 1; i < NVARS; i++)
		dc[i] += 0.5/c * g*dp[i]/uc[0];
}

inline
a_real IdealGasPhysics::getSoundSpeedFromConservedEfficiently(const a_real *const uc, 
		const a_real p) const
{
	return std::sqrt(g * p/uc[0]);
}

inline
void IdealGasPhysics::getJacobianSoundSpeedWrtConservedEfficiently(const a_real *const uc,
		const a_real p, const a_real dp[NVARS], const a_real c,
		a_real *const __restrict dc) const
{
	dc[0] += 0.5/c * g* (dp[0]*uc[0]-p)/(uc[0]*uc[0]);
	for(int i = 1; i < NVARS; i++)
		dc[i] += 0.5/c * g*dp[i]/uc[0];
}

inline
a_real IdealGasPhysics::getEntropyFromConserved(const a_real *const uc) const
{
	return getPressureFromConserved(uc)/std::pow(uc[0],g);
}

inline
void IdealGasPhysics::convertConservedToPrimitive(const a_real *const uc, a_real *const up) const
{
	up[0] = uc[0];
	a_real rhovmag2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++) {
		rhovmag2 += uc[idim]*uc[idim];
		up[idim] = uc[idim]/uc[0];
	}
	up[NDIM+1] = (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
}

inline
void IdealGasPhysics::convertConservedToPrimitive2(const a_real *const uc, a_real *const up) const
{
	up[0] = uc[0];
	a_real rhovmag2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++) {
		rhovmag2 += uc[idim]*uc[idim];
		up[idim] = uc[idim]/uc[0];
	}
	a_real p = (g-1.0)*(uc[NDIM+1] - 0.5*rhovmag2/uc[0]);
	up[NVARS-1] = g*Minf*Minf*p/uc[0];
}

inline
void IdealGasPhysics::convertPrimitiveToConserved(const a_real *const up, a_real *const uc) const
{
	uc[0] = up[0];
	a_real vmag2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++) {
		vmag2 += up[idim]*up[idim];
		uc[idim] = up[0]*up[idim];
	}
	uc[NDIM+1] = up[NDIM+1]/(g-1.0) + 0.5*up[0]*vmag2;
}

inline
void IdealGasPhysics::convertPrimitiveToPrimitive2(const a_real *const up, a_real *const up2) const
{
	a_real t = getTemperatureFromPrimitive(up);
	for(int i = 0; i < NVARS-1; i++)
		up2[i] = up[1];
	up2[NVARS-1] = t;
}

inline
a_real IdealGasPhysics::getDensityFromPressureTemperature(const a_real pressure, 
		const a_real temperature) const
{
	return g*Minf*Minf*pressure/temperature;
}

inline
a_real IdealGasPhysics::getTemperatureFromConserved(const a_real *const uc) const
{
	const a_real p = getPressureFromConserved(uc);
	return g*Minf*Minf*p/uc[0];
}

inline
void IdealGasPhysics::getJacobianTemperatureWrtConserved(const a_real *const uc, 
		a_real *const __restrict dT) const
{
	const a_real p = getPressureFromConserved(uc);
	a_real dp[NVARS]; for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc,dp);

	a_real coef = g*Minf*Minf;
	dT[0] += coef*(dp[0]*uc[0] - p)/(uc[0]*uc[0]);
	for(int i = 1; i < NVARS; i++)
		dT[i] += coef/uc[0] * dp[i];
}

inline
a_real IdealGasPhysics::getTemperatureFromPrimitive(const a_real *const up) const
{
	return g*Minf*Minf*up[NVARS-1]/up[0];
}

inline
a_real IdealGasPhysics::getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
		const a_real *const guc) const
{
	const a_real p = getPressureFromConserved(uc);
	a_real term1 = 0, term2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++)
	{
		term1 += uc[idim]*guc[idim];
		term2 += uc[idim]*uc[idim];
	}
	a_real dpdx = (g-1.0) * (guc[NDIM+1] - 0.5/(uc[0]*uc[0])*(2*uc[0]*term1 - term2*guc[0]));
	return (uc[0]*dpdx - p*guc[0])/(uc[0]*uc[0]) * g*Minf*Minf;
}

inline
a_real IdealGasPhysics::getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
		const a_real *const gup) const
{
	const a_real p = getPressureFromConserved(uc);
	return (uc[0]*gup[NDIM+1] - p*gup[0]) / (uc[0]*uc[0]) * g*Minf*Minf;
}

inline
a_real IdealGasPhysics::getGradTemperatureFromPrimitiveAndGradPrimitive(const a_real *const up,
		const a_real *const gup) const
{
	return (up[0]*gup[NDIM+1] - up[NDIM+1]*gup[0]) / (up[0]*up[0]) * g*Minf*Minf;
}

inline
void IdealGasPhysics::getGradPrimitive2FromConservedAndGradConserved(
		const a_real *const __restrict uc, const a_real *const guc, a_real *const gp) const
{
	gp[0] = guc[0];

	// velocity derivatives from momentum derivatives
	for(int i = 1; i < NDIM+1; i++)
		gp[i] = 1/guc[0] * (guc[i] - uc[i]/uc[0]*guc[0]);
	
	const a_real p = getPressureFromConserved(uc);
	
	// pressure derivative
	a_real term1 = 0, term2 = 0;
	for(int i = 1; i < NDIM+1; i++)
	{
		term1 += uc[i]*uc[i];
		term2 += uc[i]*gp[i];
	}
	term1 *= (0.5*gp[0]/uc[0]);
	const a_real dp = (g-1.0)*(guc[NVARS-1] -term1 -term2);

	// temperature
	gp[NVARS-1] = g*Minf*Minf * (uc[0]*dp - p*gp[0])/(uc[0]*uc[0]);
}

inline
a_real IdealGasPhysics::getEnergyFromPrimitive2(const a_real *const upt) const
{
	const a_real p = upt[0]*upt[NDIM+1]/(g*Minf*Minf);
	a_real vmag2 = 0;
	for(int idim = 1; idim < NDIM+1; idim++)
		vmag2 += upt[idim]*upt[idim];
	return p/(g-1.0) + 0.5*upt[0]*vmag2;
}

inline
a_real IdealGasPhysics::getViscosityCoeffFromConserved(const a_real *const uc) const
{
	a_real T = getTemperatureFromConserved(uc);
	return (1+sC/Tinf)/(T+sC/Tinf) * std::pow(T,1.5) / Reinf;
}

inline
void IdealGasPhysics::getJacobianSutherlandViscosityWrtConserved(const a_real *const uc, 
		a_real *const __restrict dmu) const
{
	a_real T = getTemperatureFromConserved(uc);
	a_real dT[NVARS]; for(int i = 0; i < NVARS; i++) dT[i] = 0;
	getJacobianTemperatureWrtConserved(uc, dT);

	a_real coef = (1.0+sC/Tinf)/Reinf;
	a_real T15 = std::pow(T,1.5), Tm15 = std::pow(T,-1.5);
	a_real denom = (T + sC/Tinf)*(T+sC/Tinf);
	// coef * pow(T,1.5) / (T + sC/Tinf)
	for(int i = 0; i < NVARS; i++)
		dmu[i] += coef* (1.5*Tm15*dT[i]*(T+sC/Tinf) - T15*dT[i])/denom;
}

inline
a_real IdealGasPhysics::getViscosityCoeffFromPrimitive2(const a_real *const up) const
{
	return (1+sC/Tinf)/(up[NVARS-1]+sC/Tinf) * std::pow(up[NVARS-1],1.5) / Reinf;
}

inline
a_real IdealGasPhysics::getConstantViscosityCoeff() const
{
	return 1.0/Reinf;
}

inline
a_real IdealGasPhysics::getThermalConductivityFromViscosity(const a_real muhat) const
{
	return muhat / (Minf*Minf*(g-1.0)*Pr);
}

inline
void IdealGasPhysics::getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
		const a_real *const dmuhat, a_real *const __restrict dkhat) const
{
	for(int k = 0; k < NVARS; k++)
		dkhat[k] = dmuhat[k]/(Minf*Minf*(g-1.0)*Pr);
}

inline
a_real IdealGasPhysics::getFreestreamPressure() const
{
	return 1.0/(g*Minf*Minf);
}

}
#endif

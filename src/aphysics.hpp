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

/// Returns a dot product computed between the first NDIM components of the two vectors.
inline a_real dimDotProduct(const a_real *const u, const a_real *const v)
{
	a_real dot = 0;
	for(int i = 0; i < NDIM; i++)
		dot += u[i]*v[i];
	return dot;
}

/// Returns the derivatives of f/g given the derivatives of f and g (for NVARS components)
/** \note The result is added to the output array dq; so prior contents will affect the outcome.
 * \note It is possible for the output array dq to point to the same location as
 * one of the input arrays. In that case, the entire array should overlap, NOT only a part of it.
 */
inline void getQuotientDerivatives(const a_real f, const a_real *const df, 
		const a_real g, const a_real *const dg, a_real *const __restrict dq)
{
	for(int i = 0; i < NVARS; i++)
		dq[i] += (df[i]*g-f*dg[i])/(g*g);
}

/// Abstract class providing analytical fluxes and their Jacobians etc
class Physics
{
public:
	/// Computes the flux vector along some direction
	virtual void getDirectionalFluxFromConserved(const a_real *const u, const a_real* const n, 
			a_real *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void getJacobianDirectionalFluxWrtConserved(const a_real *const u, 
			const a_real* const n, 
			a_real *const dfdu) const = 0;
};
	
/// Flow-physics-related computation for single-phase ideal gas
/** The non-dimensionalization assumed is from free-stream velocities and temperatues,
 * as given in section 4.14.2 of \cite matatsuka.
 * Note that several computations here depend on the exact non-dimensionalization scheme used.
 *
 * "Primitive-2" variables are density, velocities and temperature, as opposed to
 * "primitive" variables which are density, velocities and pressure.
 *
 * There are also come calculation related to the stress-velocity properties of a Newtonian fluid.
 */
class IdealGasPhysics : public Physics
{
public:
	IdealGasPhysics(const a_real _g, const a_real M_inf, 
			const a_real T_inf, const a_real Re_inf, const a_real _Pr);

	/// Computes flux in a given direction efficiently using specific data
	/** Note that this function is independent of what kind of gas it is. 
	 * \param[in] uc Vector of conserved variables
	 * \param[in] n Vector along the direction in which the flux vector is needed
	 * \param[in] vn Normal velocity (w.r.t. the the normal n above)
	 * \param[in] p Pressure
	 * \param[in|out] flux Output vector for the flux; note that any pre-existing contents
	 *   will be replaced!
	 */
	void getDirectionalFlux(const a_real *const uc, const a_real *const n,
			const a_real vn, const a_real p, a_real *const __restrict flux) const
		__attribute__((always_inline));

	/// Computes the analytical convective flux across a face oriented in some direction
	void getDirectionalFluxFromConserved(const a_real *const u, const a_real* const n, 
			a_real *const __restrict flux) const;
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void getJacobianDirectionalFluxWrtConserved(const a_real *const u, const a_real* const n, 
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

	/// Returns the non-dimensionalized free-stream pressure
	a_real getFreestreamPressure() const;
	
	/// Computes pressure from internal energy - here it's the ideal gas relation	
	a_real getPressure(const a_real internalenergy) const;

	/// Computes pressure from conserved variables
	a_real getPressureFromConserved(const a_real *const uc) const;

	/// Computes pressure gradient from conserved variables and their gradients
	a_real getGradPressureFromConservedAndGradConserved(const a_real *const uc,
		const a_real *const guc) const;
	
	/// Derivative of pressure w.r.t. conserved variables
	/** Note that the derivative is added to the second argument - the latter is not zeroed
	 * or anything.
	 */
	void getJacobianPressureWrtConserved(const a_real *const uc, a_real *const __restrict dp) const;

	/// Gives the pressure more efficiently using an additional input - the momemtum magnitude
	void getJacobianPressureWrtConserved(const a_real *const uc, const a_real rho2vmag2,
			a_real *const __restrict dp) const;

	/// Computes temperature from density and pressure - depends on non-dimensionalization.
	a_real getTemperature(const a_real rho, const a_real p) const;

	/// Computes derivatives of temperature w.r.t. either conserved, primitive or primitive2
	/** \note Derivatives can be computed w.r.t. any variable-set that 
	 * has density as the first variable; the specific variables w.r.t. which derivatives are
	 * computed depends on dp.
	 */
	void getJacobianTemperature(const a_real rho, 
			const a_real p, const a_real *const dp,
			a_real *const __restrict dT) const;
	
	/// Computes speed of sound from density and pressure
	a_real getSoundSpeed(const a_real rho, const a_real p) const;

	/// Derivative of sound speed
	/** The variable-set w.r.t. which the differentiation happens is the same as that w.r.t.
	 * which the pressure was differentiated for getting dp, similar to the case with
	 * \ref getJacobianTemperature.
	 */
	void getJacobianSoundSpeed(const a_real rho,
			const a_real p, const a_real dp[NVARS], const a_real c,
			a_real *const __restrict dc) const;

	/// Computes speed of sound from conserved variables
	a_real getSoundSpeedFromConserved(const a_real *const uc) const;

	/// Derivative of sound speed w.r.t. conserved variables
	void getJacobianSoundSpeedWrtConserved(const a_real *const uc,
			a_real *const __restrict dc) const;

	/// Derivatives of normal Mach number w.r.t. conserved variables
	void getJacobianMachNormalWrtConserved(const a_real *const uc,
		const a_real *const n,
		a_real *const __restrict dmn) const;

	/// Computes an entropy \f$ p/ \rho^\gamma \f$ from conserved variables
	a_real getEntropyFromConserved(const a_real *const uc) const;

	/// Compute energy from pressure, density and magnitude of velocity
	a_real getEnergyFromPressure(const a_real p, const a_real d, const a_real vmag2) const;

	/// Computes total energy from primitive variabes
	a_real getEnergyFromPrimitive(const a_real *const up) const;

	/// Computes total energy from a vector of density, velocities and temperature
	/** All quantities are non-dimensional.
	 */
	a_real getEnergyFromPrimitive2(const a_real *const upt) const;

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void getPrimitiveFromConserved(const a_real *const uc, a_real *const up) const;
	
	/// Convert conserved variables to primitive-2 variables; depends on non-dimensionalization
	void getPrimitive2FromConserved(const a_real *const uc, a_real *const up) const;
	
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
	void getConservedFromPrimitive(const a_real *const up, a_real *const uc) const;

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

	/// Computes non-dim temperature gradient from non-dim density, pressure and their gradients
	a_real getGradTemperature(const a_real rho, const a_real gradrho, 
		const a_real p, const a_real gradp) const;

	/// Compute non-dim temperature spatial derivative 
	/// from non-dim conserved variables and their spatial derivatives
	a_real getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
			const a_real *const guc) const;

	/// Compute spatial derivative of non-dim temperature from non-dim conserved variables and
	/// spatial derivatives of non-dim *primitive* variables
	a_real getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
			const a_real *const gup) const;

	/// Get primitive-2 gradients from conserved variables and their gradients
	/** \warning Here, the output array gp is overwritten, rather than added to.
	 */
	void getGradPrimitive2FromConservedAndGradConserved(const a_real *const __restrict uc,
			const a_real *const guc, a_real *const gp) const;
	
	/// Computes non-dimensional viscosity coeff using Sutherland's law from temperature
	/** By viscosity coefficient, we mean dynamic viscosity divided by 
	 * the free-stream Reynolds number.
	 */
	a_real getViscosityCoeffFromTemperature(const a_real T) const;

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

	/// Returns non-dimensional free-stream viscosity coefficient
	a_real getConstantViscosityCoeff() const;

	/// Computes non-dimensional conductivity from non-dimensional dynamic viscosity coeff
	a_real getThermalConductivityFromViscosity(const a_real muhat) const;

	/** \brief Computes derivatives of non-dim thermal conductivity w.r.t. conserved variables
	 * given derivatives of the non-dim viscosity coeff w.r.t. conserved variables.
	 */
	void getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
			const a_real *const dmuhat, a_real *const __restrict dkhat) const;

	/// Computes the stress tensor using gradients of primitive variables
	/** Can also use gradients of primitive-2 variables - it only uses velocity gradients.
	 * The result is assigned to the output array stress, so prior contents are lost.
	 * \param[in] mu Non-dimensional viscosity divided by free-stream Reynolds number
	 * \param[in] grad Gradients of primitve variables
	 * \param[in,out] stress Components of the stress tensor on output
	 */
	__attribute((always_inline))
	void getStressTensor(const a_real mu, const a_real grad[NDIM][NVARS], 
			a_real stress[NDIM][NDIM]) const;

	/// Computes Jacobian of stress tensor using Jacobian of gradients of primitive variables
	/** Assigns the computed Jacobian to the output array components so prior contents are lost.
	 */
	void getJacobianStress(const a_real mu, const a_real *const dmu,
		const a_real grad[NDIM][NVARS], const a_real dgrad[NDIM][NVARS][NVARS],
		a_real stress[NDIM][NDIM], a_real dstress[NDIM][NDIM][NVARS]) const;

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
void IdealGasPhysics::getDirectionalFlux(const a_real *const uc, const a_real *const n,
		const a_real vn, const a_real p, a_real *const __restrict flux) const
{
	flux[0] = vn*uc[0];
	for(int i = 1; i < NDIM+1; i++)
		flux[i] = vn*uc[i] + p*n[i-1];
	flux[NVARS-1] = vn*(uc[NVARS-1] + p);
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
a_real IdealGasPhysics::getPressure(const a_real internalenergy) const {
	return (g-1.0)*internalenergy;
}

// not restricted to ideal gases
inline
a_real IdealGasPhysics::getPressureFromConserved(const a_real *const uc) const
{
	return getPressure(uc[NDIM+1] - 0.5*dimDotProduct(&uc[1],&uc[1])/uc[0]);
}

inline 
a_real IdealGasPhysics::getGradPressureFromConservedAndGradConserved(const a_real *const uc,
		const a_real *const guc) const
{
	const a_real term1 = dimDotProduct(&uc[1],&guc[1]);
	const a_real term2 = dimDotProduct(&uc[1],&uc[1]);
	return (g-1.0) * (guc[NDIM+1] - 0.5/(uc[0]*uc[0]) * (2.0*uc[0]*term1 - term2*guc[0]));
}

inline
void IdealGasPhysics::getJacobianPressureWrtConserved(const a_real *const uc, 
		a_real *const __restrict dp) const
{
	dp[0] += (g-1.0)*0.5*dimDotProduct(&uc[1],&uc[1])/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] += -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] += (g-1.0);
}

inline
void IdealGasPhysics::getJacobianPressureWrtConserved(const a_real *const uc, 
		const a_real rho2vmag2,
		a_real *const __restrict dp) const
{
	dp[0] += (g-1.0)*0.5*rho2vmag2/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] += -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] += (g-1.0);
}

inline
a_real IdealGasPhysics::getTemperature(const a_real rho, const a_real p) const
{
  return p/rho * g*Minf*Minf;
}

inline
void IdealGasPhysics::getJacobianTemperature(const a_real rho, 
		const a_real p, const a_real *const dp,
		a_real *const __restrict dT) const
{
	const a_real coef = g*Minf*Minf;
	dT[0] += coef*(dp[0]*rho - p)/(rho*rho);
	for(int i = 1; i < NVARS; i++)
		dT[i] += coef/rho * dp[i];
}

inline
a_real IdealGasPhysics::getSoundSpeed(const a_real rho, const a_real p) const
{
	return std::sqrt(g * p/rho);
}

inline
void IdealGasPhysics::getJacobianSoundSpeed(const a_real rho,
		const a_real p, const a_real dp[NVARS], const a_real c,
		a_real *const __restrict dc) const
{
	dc[0] += 0.5/c * g* (dp[0]*rho-p)/(rho*rho);
	for(int i = 1; i < NVARS; i++)
		dc[i] += 0.5/c * g*dp[i]/rho;
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
a_real IdealGasPhysics::getSoundSpeedFromConserved(const a_real *const uc) const
{
  return getSoundSpeed(uc[0],getPressureFromConserved(uc));
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
void IdealGasPhysics::getJacobianSoundSpeedWrtConserved(const a_real *const uc,
		a_real *const __restrict dc) const
{
	a_real p =getPressureFromConserved(uc);
	a_real dp[NVARS]; 
	for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc, dp);

	const a_real c = getSoundSpeed(uc[0],p);
	getJacobianSoundSpeed(uc[0], p, dp, c, dc);
}

inline
void IdealGasPhysics::getJacobianMachNormalWrtConserved(const a_real *const uc,
		const a_real *const n,
		a_real *const __restrict dmn) const
{
	a_real dc[NVARS], dvn[NVARS];
	zeros(dc,NVARS);
	const a_real c = getSoundSpeedFromConserved(uc);
	getJacobianSoundSpeedWrtConserved(uc, dc);

	const a_real vn = dimDotProduct(&uc[1],n)/uc[0];
	dvn[0] = -vn/uc[0];
	dvn[1] = n[0]/uc[0];
	dvn[2] = n[1]/uc[0];
	dvn[3] = 0;

	getQuotientDerivatives(vn, dvn, c, dc, dmn);
}

inline
a_real IdealGasPhysics::getEntropyFromConserved(const a_real *const uc) const
{
	return getPressureFromConserved(uc)/std::pow(uc[0],g);
}

inline
a_real IdealGasPhysics::getEnergyFromPressure(const a_real p, const a_real d, const a_real vmag2) 
	const 
{
	return p/(g-1.0) + 0.5*d*vmag2;
}

// independent of non-dimensionalization
inline
a_real IdealGasPhysics::getEnergyFromPrimitive(const a_real *const up) const
{
	//return up[NVARS-1]/(g-1.0) + 0.5*up[0]*dimDotProduct(&up[1],&up[1]);
	return getEnergyFromPressure(up[NVARS-1], up[0], dimDotProduct(&up[1],&up[1]));
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
void IdealGasPhysics::getPrimitiveFromConserved(const a_real *const uc, a_real *const up) const
{
	up[0] = uc[0];
	const a_real p = getPressureFromConserved(uc);
	for(int idim = 1; idim < NDIM+1; idim++) {
		up[idim] = uc[idim]/uc[0];
	}
	up[NDIM+1] = p;
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
void IdealGasPhysics::getPrimitive2FromConserved(const a_real *const uc, a_real *const up) const
{
	up[0] = uc[0];
	const a_real p = getPressureFromConserved(uc);
	for(int idim = 1; idim < NDIM+1; idim++) {
		up[idim] = uc[idim]/uc[0];
	}
	up[NVARS-1] = getTemperature(uc[0],p);
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
void IdealGasPhysics::getConservedFromPrimitive(const a_real *const up, a_real *const uc) const
{
	uc[0] = up[0];
	const a_real rhoE = getEnergyFromPrimitive(up);
	for(int idim = 1; idim < NDIM+1; idim++) {
		uc[idim] = up[0]*up[idim];
	}
	uc[NDIM+1] = rhoE;
}

inline
a_real IdealGasPhysics::getDensityFromPressureTemperature(const a_real pressure, 
		const a_real temperature) const
{
	return g*Minf*Minf*pressure/temperature;
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
a_real IdealGasPhysics::getTemperatureFromConserved(const a_real *const uc) const
{
	return getTemperature(uc[0], getPressureFromConserved(uc));
}

// independent of non-dimensionalization
// not restricted to ideal gases
inline
void IdealGasPhysics::getJacobianTemperatureWrtConserved(const a_real *const uc, 
		a_real *const __restrict dT) const
{
	const a_real p = getPressureFromConserved(uc);
	a_real dp[NVARS]; for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc,dp);
	getJacobianTemperature(uc[0],p,dp, dT);
}

// independent of non-dimensionalization
inline
a_real IdealGasPhysics::getTemperatureFromPrimitive(const a_real *const up) const
{
	return getTemperature(up[0], up[NVARS-1]);
}

inline
a_real IdealGasPhysics::getGradTemperature(const a_real rho, const a_real gradrho, 
		const a_real p, const a_real gradp) const
{
	return (gradp*rho - p*gradrho) / (rho*rho) * g*Minf*Minf;
}

inline
a_real IdealGasPhysics::getGradTemperatureFromConservedAndGradConserved(const a_real *const uc, 
		const a_real *const guc) const
{
	const a_real p = getPressureFromConserved(uc);
	const a_real dpdx = getGradPressureFromConservedAndGradConserved(uc, guc);
	return getGradTemperature(uc[0], guc[0], p, dpdx);
}

inline
a_real IdealGasPhysics::getGradTemperatureFromConservedAndGradPrimitive(const a_real *const uc,
		const a_real *const gup) const
{
	const a_real p = getPressureFromConserved(uc);
	return getGradTemperature(uc[0], gup[0], p, gup[NVARS-1]);
}

inline
void IdealGasPhysics::getGradPrimitive2FromConservedAndGradConserved(
		const a_real *const __restrict uc, const a_real *const guc, a_real *const gup) const
{
	gup[0] = guc[0];

	// velocity derivatives from momentum derivatives
	for(int i = 1; i < NDIM+1; i++)
		gup[i] = (guc[i]*uc[0]-uc[i]*guc[0])/(uc[0]*uc[0]);
	
	const a_real p = getPressureFromConserved(uc);
	
	/* 
	 * Note that beyond this point, we assume we don't have access to momentum grads anymore,
	 * because we want to allow aliasing of guc and gp and we have modified gp above.
	 * So we only use density grad (gup[0] or guc[0]), velocity grads just computed (gup[1:NDIM+1]) 
	 * and energy grad (guc[NDIM+1]) going forward. We CANNOT use guc[1:NDIM+1].
	 */

	// pressure derivative
	a_real term1 = 0, term2 = 0;
	for(int i = 1; i < NDIM+1; i++)
	{
		term1 += uc[i]*gup[i];
		term2 += uc[i]*uc[i];
	}
	term2 *= 0.5*gup[0]/(uc[0]*uc[0]);
	const a_real dp = (g-1.0)*(guc[NVARS-1] -term2 -term1);

	// temperature
	gup[NVARS-1] = getGradTemperature(uc[0],gup[0], p, dp);
}

inline
a_real IdealGasPhysics::getEnergyFromPrimitive2(const a_real *const upt) const
{
	const a_real p = upt[0]*upt[NDIM+1]/(g*Minf*Minf);
	return p/(g-1.0) + 0.5*upt[0]*dimDotProduct(&upt[1],&upt[1]);
}

inline
a_real IdealGasPhysics::getViscosityCoeffFromTemperature(const a_real T) const
{
	return (1.0+sC/Tinf)/(T+sC/Tinf) * std::pow(T,1.5) / Reinf;
}

inline
a_real IdealGasPhysics::getViscosityCoeffFromConserved(const a_real *const uc) const
{
	const a_real T = getTemperatureFromConserved(uc);
	return getViscosityCoeffFromTemperature(T);
}

inline
void IdealGasPhysics::getJacobianSutherlandViscosityWrtConserved(const a_real *const uc, 
		a_real *const __restrict dmu) const
{
	const a_real T = getTemperatureFromConserved(uc);
	a_real dT[NVARS]; for(int i = 0; i < NVARS; i++) dT[i] = 0;
	getJacobianTemperatureWrtConserved(uc, dT);

	const a_real coef = (1.0+sC/Tinf)/Reinf;
	const a_real T15 = std::pow(T,1.5), Tm15 = std::pow(T,-1.5);
	const a_real denom = (T + sC/Tinf)*(T+sC/Tinf);
	// coef * pow(T,1.5) / (T + sC/Tinf)
	for(int i = 0; i < NVARS; i++)
		dmu[i] += coef* (1.5*Tm15*dT[i]*(T+sC/Tinf) - T15*dT[i])/denom;
}

inline
a_real IdealGasPhysics::getConstantViscosityCoeff() const {
	return 1.0/Reinf;
}

inline
a_real IdealGasPhysics::getThermalConductivityFromViscosity(const a_real muhat) const {
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
a_real IdealGasPhysics::getFreestreamPressure() const {
	return 1.0/(g*Minf*Minf);
}

inline
void IdealGasPhysics::getStressTensor(const a_real mu, const a_real grad[NDIM][NVARS], 
		a_real stress[NDIM][NDIM]) const
{
	// divergence of velocity times second viscosity
	a_real ldiv = 0;
	for(int j = 0; j < NDIM; j++)
		ldiv += grad[j][j+1];
	ldiv *= 2.0/3.0*mu;

	for(int i = 0; i < NDIM; i++) 
	{
		for(int j = 0; j < NDIM; j++)
			stress[i][j] = mu*(grad[i][j+1] + grad[j][i+1]);

		stress[i][i] -= ldiv;
	}
}

}
#endif

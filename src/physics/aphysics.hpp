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
#include "mathutils.hpp"

namespace fvens {

/// Abstract class providing analytical fluxes and their Jacobians etc
template <typename scalar>
class Physics
{
public:
	/// Computes the flux vector along some direction
	virtual void getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n, 
			scalar *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void getJacobianDirectionalFluxWrtConserved(const scalar *const u, 
			const scalar* const n, 
			scalar *const dfdu) const = 0;
};
	
/// Flow-physics-related computation for single-phase ideal gas
/** The non-dimensionalization assumed is from free-stream velocities and temperatures,
 * as given in section 4.14.2 of \cite matatsuka.
 * Note that several computations here depend on the exact non-dimensionalization scheme used.
 * Unless otherwise specified, all methods are assumed to accept non-dimensional quantities and
 * give non-dimensional output variables.
 *
 * "Primitive-2" variables are density, velocities and temperature, as opposed to
 * "primitive" variables which are density, velocities and pressure.
 *
 * There are also come calculation related to the stress-velocity properties of a Newtonian fluid.
 */
template <typename scalar>
class IdealGasPhysics : public Physics<scalar>
{
public:
	IdealGasPhysics(const a_real _g, const a_real M_inf, 
			const a_real T_inf, const a_real Re_inf, const a_real _Pr);

	/// Returns an array containing the non-dimensional free-stream state
	/** \param aoa The angle of attack in radians
	 */
	std::array<scalar,NVARS> compute_freestream_state(const a_real aoa) const;

	/// Computes flux in a given direction efficiently using specific data
	/** Note that this function is independent of what kind of gas it is. 
	 * \param[in] uc Vector of conserved variables
	 * \param[in] n Vector along the direction in which the flux vector is needed
	 * \param[in] vn Normal velocity (w.r.t. the the normal n above)
	 * \param[in] p Pressure
	 * \param[in|out] flux Output vector for the flux; note that any pre-existing contents
	 *   will be replaced!
	 */
	void getDirectionalFlux(const scalar *const uc, const scalar *const n,
			const scalar vn, const scalar p, scalar *const __restrict flux) const
		__attribute__((always_inline));

	/// Computes the analytical convective flux across a face oriented in some direction
	void getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n, 
			scalar *const __restrict flux) const;
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void getJacobianDirectionalFluxWrtConserved(const scalar *const u, const scalar* const n, 
			scalar *const __restrict dfdu) const;

	/// Computes derivatives of the squared velocity magnitude w.r.t. conserved variables
	/** \warning uc and dvmag2 must not point to the same memory locations.
	 */
	void getJacobianVmag2WrtConserved(const scalar *const uc, 
		scalar *const __restrict dvmag2) const;

	/// Outputs various quantities, especially needed by numerical fluxes
	/** \param[in] uc Conserved variables
	 * \param[in] n Normal vector
	 * \param[out] v Velocity vector (of length NDIM)
	 * \param[out] vn Normal velocity
	 * \param[out] vm2 Square of velocity magnitude
	 * \param[out] p Pressure
	 * \param[out] H Specific enthalpy
	 */
	void getVarsFromConserved(const scalar *const uc, const scalar *const n,
			scalar *const v,
			scalar& vn,
			scalar& p, scalar& H ) const
		__attribute((always_inline));

	/// Computes derivatives of variables computed in getVarsFromConserved
	/** \param[in] uc Vector of conserved variables
	 * \param[in] n Normal vector
	 * \param[in,out] dv is a NDIM x NVARS matrix stored as a row-major 1D array
	 * \warning The result is added to the original content of the  output variables!
	 */
	void getJacobianVarsWrtConserved(const scalar *const uc, const scalar *const n,
		scalar *const dv, scalar dvn[NVARS],
		scalar dp[NVARS], scalar dH[NVARS]) const;

	/// Returns the non-dimensionalized free-stream pressure
	scalar getFreestreamPressure() const;
	
	/// Computes pressure from internal energy - here it's the ideal gas relation	
	scalar getPressure(const scalar internalenergy) const;

	/// Computes pressure from conserved variables
	scalar getPressureFromConserved(const scalar *const uc) const;

	/// Computes pressure gradient from conserved variables and their gradients
	scalar getGradPressureFromConservedAndGradConserved(const scalar *const uc,
		const scalar *const guc) const;
	
	/// Derivative of pressure w.r.t. conserved variables
	/** Note that the derivative is added to the second argument - the latter is not zeroed
	 * or anything.
	 */
	void getJacobianPressureWrtConserved(const scalar *const uc, scalar *const __restrict dp) const
		__attribute((always_inline));

	/// Gives the pressure more efficiently using an additional input - the momemtum magnitude
	void getJacobianPressureWrtConserved(const scalar *const uc, const scalar rho2vmag2,
			scalar *const __restrict dp) const;

	/// Computes temperature from density and pressure - depends on non-dimensionalization.
	scalar getTemperature(const scalar rho, const scalar p) const;

	/// Computes derivatives of temperature w.r.t. either conserved, primitive or primitive2
	/** \note Derivatives can be computed w.r.t. any variable-set that 
	 * has density as the first variable; the specific variables w.r.t. which derivatives are
	 * computed depends on dp.
	 */
	void getJacobianTemperature(const scalar rho, 
			const scalar p, const scalar *const dp,
			scalar *const __restrict dT) const;
	
	/// Computes speed of sound from density and pressure
	scalar getSoundSpeed(const scalar rho, const scalar p) const;

	/// Derivative of sound speed
	/** The variable-set w.r.t. which the differentiation happens is the same as that w.r.t.
	 * which the pressure was differentiated for getting dp, similar to the case with
	 * \ref getJacobianTemperature.
	 */
	void getJacobianSoundSpeed(const scalar rho,
			const scalar p, const scalar dp[NVARS], const scalar c,
			scalar *const __restrict dc) const;

	/// Computes speed of sound from conserved variables
	scalar getSoundSpeedFromConserved(const scalar *const uc) const;

	/// Derivative of sound speed w.r.t. conserved variables
	void getJacobianSoundSpeedWrtConserved(const scalar *const uc,
			scalar *const __restrict dc) const;

	/// Derivatives of normal Mach number w.r.t. conserved variables
	void getJacobianMachNormalWrtConserved(const scalar *const uc,
		const scalar *const n,
		scalar *const __restrict dmn) const;

	/// Computes an entropy \f$ p/ \rho^\gamma \f$ from conserved variables
	scalar getEntropyFromConserved(const scalar *const uc) const;

	/// Compute energy from pressure, density and square of magnitude of velocity
	scalar getEnergyFromPressure(const scalar p, const scalar d, const scalar vmag2) const;
	
	/// Compute energy from temperature, density and square of magnitude of velocity
	scalar getEnergyFromTemperature(const scalar T, const scalar d, const scalar vmag2) const;
	
	/// Computes derivatives of total energy from derivatives of temperature and |v|^2
	/** Can compute the derivatives w.r.t. any variable-set as long as
	 * the first variable is density. That variable set is decided by what variables the
	 * T and |v|^2 were differentiated with respect to.
	 * \param[in,out] drhoE The derivative of the total energy per unit volume is *added to* this
	 */
	void getJacobianEnergyFromJacobiansTemperatureVmag2(
		const scalar T, const scalar d, const scalar vmag2,
		const scalar *const dT, const scalar *const dvmag2,
		scalar *const drhoE) const;

	/// Computes total energy from primitive variabes
	scalar getEnergyFromPrimitive(const scalar *const up) const;

	/// Computes total energy from a vector of density, velocities and temperature
	/** All quantities are non-dimensional.
	 */
	scalar getEnergyFromPrimitive2(const scalar *const upt) const;

	/// Convert conserved variables to primitive variables (density, velocities, pressure)
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void getPrimitiveFromConserved(const scalar *const uc, scalar *const up) const;
	
	/// Convert conserved variables to primitive-2 variables; depends on non-dimensionalization
	void getPrimitive2FromConserved(const scalar *const uc, scalar *const up) const;
	
	/// Computes the Jacobian matrix of the conserved-to-primitive-2 transformation
	/** \f$ \partial \mathbf{u}_{prim2} / \partial \mathbf{u}_{cons} \f$. 
	 * The output is stored as 1D rowmajor.
	 *
	 * \warning The Jacobian is *added* to the output jac. If it has garbage at the outset,
	 * it will contain garbage at the end.
	 */
	void getJacobianPrimitive2WrtConserved(const scalar *const uc, 
			scalar *const __restrict jac) const;

	/// Convert primitive variables to conserved
	/** The input pointers are not assumed restricted, so the two parameters can point to
	 * the same storage.
	 */
	void getConservedFromPrimitive(const scalar *const up, scalar *const uc) const;

	/// Computes density from pressure and temperature using ideal gas relation;
	/// All quantities are non-dimensional
	scalar getDensityFromPressureTemperature(const scalar pressure, const scalar temperature) const;

	/// Computes derivatives of density from derivatives of pressure and temperature
	void getJacobianDensityFromJacobiansPressureTemperature(
			const scalar pressure, const scalar temperature,
			const scalar *const dp, const scalar *const dT, scalar *const drho) const;

	/// Computes non-dimensional temperature from non-dimensional conserved variables
	/** \sa IdealGasPhysics
	 */
	scalar getTemperatureFromConserved(const scalar *const uc) const;
	
	/// Computes derivatives of temperature w.r.t. conserved variables
	/** \param[in|out] dT The derivatives are added to dT, which is not zeroed initially.
	 */
	void getJacobianTemperatureWrtConserved(const scalar *const uc, 
			scalar *const __restrict dT) const;

	/// Computes non-dimensional temperature from non-dimensional primitive variables
	scalar getTemperatureFromPrimitive(const scalar *const up) const;

	/// Computes non-dim temperature gradient from non-dim density, pressure and their gradients
	scalar getGradTemperature(const scalar rho, const scalar gradrho, 
		const scalar p, const scalar gradp) const;

	/// Compute non-dim temperature spatial derivative 
	/// from non-dim conserved variables and their spatial derivatives
	scalar getGradTemperatureFromConservedAndGradConserved(const scalar *const uc, 
			const scalar *const guc) const;

	/// Compute spatial derivative of non-dim temperature from non-dim conserved variables and
	/// spatial derivatives of non-dim *primitive* variables
	scalar getGradTemperatureFromConservedAndGradPrimitive(const scalar *const uc,
			const scalar *const gup) const;

	/// Get primitive-2 gradients from conserved variables and their gradients
	/** \warning Here, the output array gp is overwritten, rather than added to.
	 */
	void getGradPrimitive2FromConservedAndGradConserved(const scalar *const __restrict uc,
			const scalar *const guc, scalar *const gp) const;
	
	/// Computes non-dimensional viscosity coeff using Sutherland's law from temperature
	/** By viscosity coefficient, we mean dynamic viscosity divided by 
	 * the free-stream Reynolds number.
	 */
	scalar getViscosityCoeffFromTemperature(const scalar T) const;

	/// Computes non-dimensional viscosity coeff using Sutherland's law from conserved variables
	/** This is the dynamic viscosity divided by the Reynolds number
	 * when non-dimensionalized as stated in IdealGasPhysics.
	 * Note that divergence terms must still be multiplied further by -2/3 
	 * and diagonal stress terms by 2.
	 */
	scalar getViscosityCoeffFromConserved(const scalar *const uc) const;

	/// Computes derivatives of Sutherland's dynamic viscosity coeff w.r.t. conserved variables
	void getJacobianSutherlandViscosityWrtConserved(const scalar *const uc, 
			scalar *const __restrict dmu) const;

	/// Returns non-dimensional free-stream viscosity coefficient
	scalar getConstantViscosityCoeff() const;

	/// Computes non-dimensional conductivity from non-dimensional dynamic viscosity coeff
	scalar getThermalConductivityFromViscosity(const scalar muhat) const;

	/** \brief Computes derivatives of non-dim thermal conductivity w.r.t. conserved variables
	 * given derivatives of the non-dim viscosity coeff w.r.t. conserved variables.
	 */
	void getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
			const scalar *const dmuhat, scalar *const __restrict dkhat) const;

	/// Computes the stress tensor using gradients of primitive variables
	/** Can also use gradients of primitive-2 variables - it only uses velocity gradients.
	 * The result is assigned to the output array stress, so prior contents are lost.
	 * \param[in] mu Non-dimensional viscosity divided by free-stream Reynolds number
	 * \param[in] grad Gradients of primitve variables
	 * \param[in,out] stress Components of the stress tensor on output
	 */
	void getStressTensor(const scalar mu, const scalar grad[NDIM][NVARS], 
			scalar stress[NDIM][NDIM]) const __attribute((always_inline));

	/// Computes Jacobian of stress tensor using Jacobian of gradients of primitive variables
	/** Assigns the computed Jacobian to the output array components so prior contents are lost.
	 */
	void getJacobianStress(const scalar mu, const scalar *const dmu,
		const scalar grad[NDIM][NVARS], const scalar dgrad[NDIM][NVARS][NVARS],
		scalar stress[NDIM][NDIM], scalar dstress[NDIM][NDIM][NVARS]) const;

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

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getDirectionalFlux(const scalar *const uc, const scalar *const n,
		const scalar vn, const scalar p, scalar *const __restrict flux) const
{
	flux[0] = vn*uc[0];
	for(int i = 1; i < NDIM+1; i++)
		flux[i] = vn*uc[i] + p*n[i-1];
	flux[NVARS-1] = vn*(uc[NVARS-1] + p);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getVarsFromConserved(const scalar *const uc, const scalar *const n,
		scalar *const __restrict v,
		scalar& vn,
		scalar& p, scalar& H ) const
{
	for(int j = 0; j < NDIM; j++)
		v[j] = uc[j+1]/uc[0]; 
	vn = dimDotProduct(v,n);
	const scalar vmag2 = dimDotProduct(v,v);
	p = (g-1.0)*(uc[3] - 0.5*uc[0]*vmag2);
	H = (uc[3]+p)/uc[0];
}

// not restricted to anything
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianVmag2WrtConserved(const scalar *const uc, 
		scalar *const __restrict dvmag2) const
{
	dvmag2[0] += -2.0/(uc[0]*uc[0]*uc[0])*dimDotProduct(&uc[1],&uc[1]);
	for(int i = 1; i < NDIM+1; i++)
		dvmag2[i] += 2.0*uc[i]/(uc[0]*uc[0]);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getPressure(const scalar internalenergy) const {
	return (g-1.0)*internalenergy;
}

// not restricted to ideal gases
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getPressureFromConserved(const scalar *const uc) const
{
	return getPressure(uc[NDIM+1] - 0.5*dimDotProduct(&uc[1],&uc[1])/uc[0]);
}

template <typename scalar>
inline 
scalar IdealGasPhysics<scalar>::getGradPressureFromConservedAndGradConserved(const scalar *const uc,
		const scalar *const guc) const
{
	const scalar term1 = dimDotProduct(&uc[1],&guc[1]);
	const scalar term2 = dimDotProduct(&uc[1],&uc[1]);
	return (g-1.0) * (guc[NDIM+1] - 0.5/(uc[0]*uc[0]) * (2.0*uc[0]*term1 - term2*guc[0]));
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianPressureWrtConserved(const scalar *const uc, 
		scalar *const __restrict dp) const
{
	dp[0] += (g-1.0)*0.5*dimDotProduct(&uc[1],&uc[1])/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] += -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] += (g-1.0);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianPressureWrtConserved(const scalar *const uc, 
		const scalar rho2vmag2,
		scalar *const __restrict dp) const
{
	dp[0] += (g-1.0)*0.5*rho2vmag2/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] += -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] += (g-1.0);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getTemperature(const scalar rho, const scalar p) const
{
  return p/rho * g*Minf*Minf;
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianTemperature(const scalar rho, 
		const scalar p, const scalar *const dp,
		scalar *const __restrict dT) const
{
	const scalar coef = g*Minf*Minf;
	dT[0] += coef*(dp[0]*rho - p)/(rho*rho);
	for(int i = 1; i < NVARS; i++)
		dT[i] += coef/rho * dp[i];
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getSoundSpeed(const scalar rho, const scalar p) const
{
	return std::sqrt(g * p/rho);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianSoundSpeed(const scalar rho,
		const scalar p, const scalar dp[NVARS], const scalar c,
		scalar *const __restrict dc) const
{
	dc[0] += 0.5/c * g* (dp[0]*rho-p)/(rho*rho);
	for(int i = 1; i < NVARS; i++)
		dc[i] += 0.5/c * g*dp[i]/rho;
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getSoundSpeedFromConserved(const scalar *const uc) const
{
  return getSoundSpeed(uc[0],getPressureFromConserved(uc));
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianSoundSpeedWrtConserved(const scalar *const uc,
		scalar *const __restrict dc) const
{
	scalar p =getPressureFromConserved(uc);
	scalar dp[NVARS]; 
	for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc, dp);

	const scalar c = getSoundSpeed(uc[0],p);
	getJacobianSoundSpeed(uc[0], p, dp, c, dc);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianMachNormalWrtConserved(const scalar *const uc,
		const scalar *const n,
		scalar *const __restrict dmn) const
{
	scalar dc[NVARS], dvn[NVARS];
	zeros(dc,NVARS);
	const scalar c = getSoundSpeedFromConserved(uc);
	getJacobianSoundSpeedWrtConserved(uc, dc);

	const scalar vn = dimDotProduct(&uc[1],n)/uc[0];
	dvn[0] = -vn/uc[0];
	dvn[1] = n[0]/uc[0];
	dvn[2] = n[1]/uc[0];
	dvn[3] = 0;

	getQuotientDerivatives(vn, dvn, c, dc, dmn);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEntropyFromConserved(const scalar *const uc) const
{
	return getPressureFromConserved(uc)/std::pow(uc[0],g);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEnergyFromPressure(const scalar p, const scalar d, const scalar vmag2) 
	const 
{
	return p/(g-1.0) + 0.5*d*vmag2;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEnergyFromTemperature(const scalar T, const scalar d, 
		const scalar vmag2) const
{
	return d * (T/(g*(g-1.0)*Minf*Minf) + 0.5*vmag2);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianEnergyFromJacobiansTemperatureVmag2(
		const scalar T, const scalar d, const scalar vmag2,
		const scalar *const dT, const scalar *const dvmag2,
		scalar *const de) const
{
	const scalar coeff = 1.0/(g*(g-1.0)*Minf*Minf);
	de[0] += coeff * (T+d*dT[0]) + 0.5 * (vmag2+d*dvmag2[0]);
	for(int i = 1; i < NVARS; i++)
		de[i] += d * (coeff*dT[i] + 0.5*dvmag2[i]);
}

// independent of non-dimensionalization
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEnergyFromPrimitive(const scalar *const up) const
{
	return getEnergyFromPressure(up[NVARS-1], up[0], dimDotProduct(&up[1],&up[1]));
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEnergyFromPrimitive2(const scalar *const upt) const
{
	/*const scalar p = upt[0]*upt[NDIM+1]/(g*Minf*Minf);
	return p/(g-1.0) + 0.5*upt[0]*dimDotProduct(&upt[1],&upt[1]);*/
	return getEnergyFromTemperature(upt[NVARS-1], upt[0], dimDotProduct(&upt[1],&upt[1]));
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getPrimitiveFromConserved(const scalar *const uc, scalar *const up) const
{
	up[0] = uc[0];
	const scalar p = getPressureFromConserved(uc);
	for(int idim = 1; idim < NDIM+1; idim++) {
		up[idim] = uc[idim]/uc[0];
	}
	up[NDIM+1] = p;
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getPrimitive2FromConserved(const scalar *const uc, scalar *const up) const
{
	up[0] = uc[0];
	const scalar p = getPressureFromConserved(uc);
	for(int idim = 1; idim < NDIM+1; idim++) {
		up[idim] = uc[idim]/uc[0];
	}
	up[NVARS-1] = getTemperature(uc[0],p);
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getConservedFromPrimitive(const scalar *const up, scalar *const uc) const
{
	uc[0] = up[0];
	const scalar rhoE = getEnergyFromPrimitive(up);
	for(int idim = 1; idim < NDIM+1; idim++) {
		uc[idim] = up[0]*up[idim];
	}
	uc[NDIM+1] = rhoE;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getDensityFromPressureTemperature(const scalar pressure, 
		const scalar temperature) const
{
	return g*Minf*Minf*pressure/temperature;
}
	
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianDensityFromJacobiansPressureTemperature(
		const scalar pressure, const scalar temperature,
		const scalar *const dp, const scalar *const dT, scalar *const drho) const
{
	for(int i = 0; i < NVARS; i++)
		drho[i] += g*Minf*Minf*(dp[i]*temperature-pressure*dT[i])/(temperature*temperature);
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getTemperatureFromConserved(const scalar *const uc) const
{
	return getTemperature(uc[0], getPressureFromConserved(uc));
}

// independent of non-dimensionalization
// not restricted to ideal gases
template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianTemperatureWrtConserved(const scalar *const uc, 
		scalar *const __restrict dT) const
{
	const scalar p = getPressureFromConserved(uc);
	scalar dp[NVARS]; for(int i = 0; i < NVARS; i++) dp[i] = 0;
	getJacobianPressureWrtConserved(uc,dp);
	getJacobianTemperature(uc[0],p,dp, dT);
}

// independent of non-dimensionalization
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getTemperatureFromPrimitive(const scalar *const up) const
{
	return getTemperature(up[0], up[NVARS-1]);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getGradTemperature(const scalar rho, const scalar gradrho, 
		const scalar p, const scalar gradp) const
{
	return (gradp*rho - p*gradrho) / (rho*rho) * g*Minf*Minf;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getGradTemperatureFromConservedAndGradConserved(const scalar *const uc, 
		const scalar *const guc) const
{
	const scalar p = getPressureFromConserved(uc);
	const scalar dpdx = getGradPressureFromConservedAndGradConserved(uc, guc);
	return getGradTemperature(uc[0], guc[0], p, dpdx);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getGradTemperatureFromConservedAndGradPrimitive(const scalar *const uc,
		const scalar *const gup) const
{
	const scalar p = getPressureFromConserved(uc);
	return getGradTemperature(uc[0], gup[0], p, gup[NVARS-1]);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getGradPrimitive2FromConservedAndGradConserved(
		const scalar *const __restrict uc, const scalar *const guc, scalar *const gup) const
{
	gup[0] = guc[0];

	// velocity derivatives from momentum derivatives
	for(int i = 1; i < NDIM+1; i++)
		gup[i] = (guc[i]*uc[0]-uc[i]*guc[0])/(uc[0]*uc[0]);
	
	const scalar p = getPressureFromConserved(uc);
	
	/* 
	 * Note that beyond this point, we assume we don't have access to momentum grads anymore,
	 * because we want to allow aliasing of guc and gp and we have modified gp above.
	 * So we only use density grad (gup[0] or guc[0]), velocity grads just computed (gup[1:NDIM+1]) 
	 * and energy grad (guc[NDIM+1]) going forward. We CANNOT use guc[1:NDIM+1].
	 */

	// pressure derivative
	scalar term1 = 0, term2 = 0;
	for(int i = 1; i < NDIM+1; i++)
	{
		term1 += uc[i]*gup[i];
		term2 += uc[i]*uc[i];
	}
	term2 *= 0.5*gup[0]/(uc[0]*uc[0]);
	const scalar dp = (g-1.0)*(guc[NVARS-1] -term2 -term1);

	// temperature
	gup[NVARS-1] = getGradTemperature(uc[0],gup[0], p, dp);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getViscosityCoeffFromTemperature(const scalar T) const
{
	return (1.0+sC/Tinf)/(T+sC/Tinf) * std::pow(T,1.5) / Reinf;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getViscosityCoeffFromConserved(const scalar *const uc) const
{
	const scalar T = getTemperatureFromConserved(uc);
	return getViscosityCoeffFromTemperature(T);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianSutherlandViscosityWrtConserved(const scalar *const uc, 
		scalar *const __restrict dmu) const
{
	const scalar T = getTemperatureFromConserved(uc);
	scalar dT[NVARS]; for(int i = 0; i < NVARS; i++) dT[i] = 0;
	getJacobianTemperatureWrtConserved(uc, dT);

	const scalar coef = (1.0+sC/Tinf)/Reinf;
	const scalar T15 = std::pow(T,1.5), Tm15 = std::pow(T,-1.5);
	const scalar denom = (T + sC/Tinf)*(T+sC/Tinf);
	// coef * pow(T,1.5) / (T + sC/Tinf)
	for(int i = 0; i < NVARS; i++)
		dmu[i] += coef* (1.5*Tm15*dT[i]*(T+sC/Tinf) - T15*dT[i])/denom;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getConstantViscosityCoeff() const {
	return 1.0/Reinf;
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getThermalConductivityFromViscosity(const scalar muhat) const {
	return muhat / (Minf*Minf*(g-1.0)*Pr);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(
		const scalar *const dmuhat, scalar *const __restrict dkhat) const 
{
	for(int k = 0; k < NVARS; k++)
		dkhat[k] = dmuhat[k]/(Minf*Minf*(g-1.0)*Pr);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getFreestreamPressure() const {
	return (scalar)(1.0/(g*Minf*Minf));
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getStressTensor(const scalar mu, const scalar grad[NDIM][NVARS], 
		scalar stress[NDIM][NDIM]) const
{
	// divergence of velocity times second viscosity
	scalar ldiv = 0;
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

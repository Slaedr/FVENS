/** \file
 * \brief Definitions of inline physics functions
 * \author Aditya Kashi
 */

#ifndef APHYSICS_DEFS_H
#define APHYSICS_DEFS_H

#include "aphysics.hpp"

namespace fvens {

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getDirectionalFlux(const scalar *const uc, const scalar *const n,
                                                 const scalar vn, const scalar p,
                                                 scalar *const __restrict flux) const
{
	flux[0] = vn*uc[0];
	for(int i = 1; i < NDIM+1; i++)
		flux[i] = vn*uc[i] + p*n[i-1];
	flux[NDIM+1] = vn*(uc[NDIM+1] + p);
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
scalar IdealGasPhysics<scalar>::getDeltaPressureFromConserved(const scalar *const u,
                                                              const scalar *const du) const
{
	scalar unew[nvars];
	for(int i = 0; i < nvars; i++)
		unew[i] = u[i] + du[i];

	scalar dp = 0;
	for(int i = 2; i < NDIM+2; i++) {
		dp -= ((u[i]+unew[i])*(u[0]+unew[0])/2.0*du[i]
		       - (unew[i]*unew[i]+u[i]*u[i])/2.0*du[0]);
	}
	dp = (g-1.0)*(du[nvars-1] - 1.0/(2*u[0]*unew[0])*dp);
	return dp;
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
	dp[0] = (g-1.0)*0.5*dimDotProduct(&uc[1],&uc[1])/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] = -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] = (g-1.0);
}

template <typename scalar>
inline
void IdealGasPhysics<scalar>::getJacobianPressureWrtConserved(const scalar *const uc,
		const scalar rho2vmag2,
		scalar *const __restrict dp) const
{
	dp[0] = (g-1.0)*0.5*rho2vmag2/(uc[0]*uc[0]);
	for(int i = 1; i < NDIM+1; i++)
		dp[i] = -(g-1.0)*uc[i]/uc[0];
	dp[NDIM+1] = (g-1.0);
}

// Depends on non-dimensionalization
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getTemperature(const scalar rho, const scalar p) const
{
  return p/rho * g*Minf*Minf;
}

// Depends on non-dimensionalization
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

// independent of non-dimensionalization
template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getSoundSpeed(const scalar rho, const scalar p) const
{
	return sqrt(g * p/rho);
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
	for(int i = 0; i < NVARS; i++)
		dp[i] = 0;
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
	for(int i = 1; i < NDIM+1; i++)
		dvn[i] = n[i-1]/uc[0];
	dvn[NDIM+1] = 0;

	getQuotientDerivatives(vn, dvn, c, dc, dmn);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getEntropyFromConserved(const scalar *const uc) const
{
	return getPressureFromConserved(uc)/pow(uc[0],g);
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
	scalar dp[NVARS];
	for(int i = 0; i < NVARS; i++)
		dp[i] = 0;
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
	return (1.0+sC/Tinf)/(T+sC/Tinf) * pow(T,1.5) / Reinf;
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
	scalar dT[NVARS];
	for(int i = 0; i < NVARS; i++)
		dT[i] = 0;
	getJacobianTemperatureWrtConserved(uc, dT);

	const scalar coef = (1.0+sC/Tinf)/Reinf;
	const scalar T15 = pow(T,1.5), Tm15 = pow(T,-1.5);
	const scalar denom = (T + sC/Tinf)*(T+sC/Tinf);
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
void IdealGasPhysics<scalar>::
getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(const scalar *const dmuhat,
                                                                   scalar *const __restrict dkhat) const
{
	for(int k = 0; k < NVARS; k++)
		dkhat[k] = dmuhat[k]/(Minf*Minf*(g-1.0)*Pr);
}

template <typename scalar>
inline
scalar IdealGasPhysics<scalar>::getFreestreamPressure() const {
	return (1.0/(g*Minf*Minf));
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

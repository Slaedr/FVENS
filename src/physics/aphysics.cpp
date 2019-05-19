/** \file aphysics.cpp
 * \brief Implementation of analytical flux computation and variable conversions
 * \author Aditya Kashi
 * \date 2017 September 13
 */

#include "aphysics_defs.hpp"
#include <iostream>
#include "utilities/adolcutils.hpp"

namespace fvens {

template <typename scalar>
Physics<scalar>::~Physics() { }

template <typename scalar>
IdealGasPhysics<scalar>::IdealGasPhysics(const freal _g, const freal M_inf,
		const freal T_inf, const freal Re_inf, const freal _Pr)
	: g{_g}, Minf{M_inf}, Tinf{T_inf}, Reinf{Re_inf}, Pr{_Pr}, sC{110.5}
{
#ifdef DEBUG
	// std::cout << " IdealGasPhysics: Physical parameters:\n";
	// std::cout << "  Adiabatic index = " <<g << ", M_infty = " <<Minf << ", T_infty = " << Tinf
	// 	<< "\n   Re_infty = " << Reinf << ", Pr = " << Pr << std::endl;
#endif
}

template <typename scalar>
void IdealGasPhysics<scalar>::getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n,
                                                              scalar *const __restrict flux) const
{
	const scalar vn = dimDotProduct(&u[1],n)/u[0];
	const scalar p = getPressure(u[NDIM+1] - 0.5*dimDotProduct(&u[1],&u[1])/u[0]);
	getDirectionalFlux(u, n, vn, p, flux);
}

/** The reference density and reference velocity are the free-stream values, therefore
 * the non-dimensional free-stream density and velocity magnitude are 1.0.
 * Assume x is the roll axis, y is the yaw axis and z is the pitch axis.
 * The angle beta is the angle the wind makes with its projection on the x-y plane.
 * Then A.o.A. is the angle that the wind's projection on the x-y plane makes with the x axis.
 */
template <typename scalar>
std::array<freal,NVARS> IdealGasPhysics<scalar>::compute_freestream_state(const freal aoa) const
{
	std::array<freal,nvars> uinf;
	const freal beta = 0;
	uinf[0] = 1.0;
	uinf[1] = cos(aoa)*cos(beta);
	uinf[2] = sin(aoa)*cos(beta);
	if(NDIM == 3)
		uinf[3] = sin(beta);
    uinf[NDIM+1] = getvalue<scalar>(getEnergyFromPressure(getFreestreamPressure(),1.0,1.0));
	return uinf;
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianDirectionalFluxWrtConserved(const scalar *const u,
                                                                     const scalar* const n,
                                                                     scalar *const __restrict dfdu) const
{
#if 0
	const scalar rhovn = dimDotProduct(&u[1],n), u02 = u[0]*u[0];
	// first row
	dfdu[0] = 0;
	dfdu[1] = n[0];
	dfdu[2] = n[1];
	dfdu[3] = 0;

	// second row
	dfdu[4] = (-rhovn*u[1] + (g-1)*n[0]*(dimDotProduct(&u[1],&u[1]))/2.0) / u02;
	dfdu[5] = ((3.0-g)*u[1]*n[0] + u[2]*n[1]) / u[0];
	dfdu[6] = (n[1]*u[1]-(g-1)*n[0]*u[2])/u[0];
	dfdu[7] = (g-1)*n[0];
	// third row
	dfdu[8] = (-rhovn*u[2] + (g-1)*n[1]*(dimDotProduct(&u[1],&u[1]))/2.0) / u02;
	dfdu[9] = (n[0]*u[2]-(g-1)*n[1]*u[1])/u[0];
	dfdu[10]= ((3.0-g)*u[2]*n[1]+u[1]*n[0])/u[0];
	dfdu[11]= (g-1)*n[1];

	// fourth row
	dfdu[12]= rhovn*((g-1) * (dimDotProduct(&u[1],&u[1]))/(u02*u[0]) - g*u[3]/u02);
	dfdu[13]= g*u[3]*n[0]/u[0] - (g-1)*0.5/u02*(3*u[1]*u[1]*n[0]+u[2]*u[2]*n[0]+2*u[1]*u[2]*n[1]);
	dfdu[14]= g*u[3]*n[1]/u[0] - (g-1)*0.5/u02*(2*u[1]*u[2]*n[0]+u[1]*u[1]*n[1]+3*u[2]*u[2]*n[1]);
	dfdu[15]= g*rhovn/u[0];
#endif

	//flux[0] = vn*uc[0];
	dfdu[0] = 0;
	for(int i = 1; i < NDIM+1; i++)
		dfdu[i] = n[i-1];
	dfdu[NDIM+1] = 0;

	const freal p = getPressureFromConserved(u);
	freal dp[nvars];
	getJacobianPressureWrtConserved(u, dp);

	const freal vn = dimDotProduct(&u[1],n)/u[0];
	freal dvn[nvars];
	dvn[0] = -vn/u[0];
	for(int i = 1; i < NDIM+1; i++)
		dvn[i] = n[i-1]/u[0];
	dvn[NDIM+1] = 0;

	for(int i = 1; i < NDIM+1; i++)
	{
		//flux[i] = vn*uc[i] + p*n[i-1];
		dfdu[i*nvars] = -vn*u[i]/u[0] + dp[0]*n[i-1];
		for(int j = 1; j < NDIM+1; j++)
		{
			if(i == j)
				dfdu[i*nvars+j] = dvn[j]*u[i] + vn + dp[j]*n[i-1];
			else
				dfdu[i*nvars+j] = dvn[j]*u[i] + dp[j]*n[i-1];
		}
		dfdu[i*nvars+NDIM+1] = dp[NDIM+1]*n[i-1];
	}

	//flux[nvars-1] = vn*(uc[nvars-1] + p);
	dfdu[(NDIM+1)*nvars] = -vn/u[0]*(u[NDIM+1]+p) + vn*dp[0];
	for(int j = 1; j < NDIM+1; j++)
		dfdu[(NDIM+1)*nvars+j] = n[j-1]/u[0]*(u[NDIM+1]+p) + vn*dp[j];
	dfdu[(NDIM+1)*nvars+NDIM+1] = vn*(1.0 + dp[NDIM+1]);
}

template <typename scalar>
void IdealGasPhysics<scalar>
::getJacobianVarsWrtConserved(const scalar *const uc, const scalar *const n,
                              scalar *const __restrict dv, scalar *const __restrict dvn,
                              scalar *const __restrict dp, scalar *const __restrict dH) const
{
	for(int j = 0; j < NDIM; j++)
	{
		dv[j*nvars+0] += -uc[j+1]/(uc[0]*uc[0]);
		dv[j*nvars+j+1] += 1.0/uc[0];
	}

	for(int j = 0; j < NDIM; j++) {
		dvn[0] += dv[j*nvars]*n[j];
		dvn[j+1] += n[j]/uc[0];
	}

	const scalar p = getPressureFromConserved(uc);
	getJacobianPressureWrtConserved(uc, dp);

	dH[0] += (dp[0]*uc[0] - (uc[NDIM+1]+p))/(uc[0]*uc[0]);
	for(int j = 1; j < NDIM+1; j++)
		dH[j] += dp[j]/uc[0];
	dH[3] += (1.0+dp[NDIM+1])/uc[0];
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianPrimitive2WrtConserved(const scalar *const uc,
		scalar *const __restrict jac) const
{
	jac[0] += 1.0;

	const scalar rho2vmag2 = dimDotProduct(&uc[1],&uc[1]);

	for(int idim = 1; idim < NDIM+1; idim++) {
		// d(up[idim])
		jac[idim*nvars+0] += -uc[idim]/(uc[0]*uc[0]);
		jac[idim*nvars+idim] += 1.0/uc[0];
	}

	const scalar p = getPressure(uc[NDIM+1] - 0.5*rho2vmag2/uc[0]);
	scalar dp[nvars];
	zeros(dp, nvars);
	getJacobianPressureWrtConserved(uc, rho2vmag2, dp);

	getJacobianTemperature(uc[0], p, dp, &jac[(NDIM+1)*nvars]);
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianStress(const scalar mu, const scalar *const dmu,
                                                const scalar grad[NDIM][nvars],
                                                const scalar dgrad[NDIM][nvars][nvars],
                                                scalar stress[NDIM][NDIM],
                                                scalar dstress[NDIM][NDIM][nvars]) const
{
	scalar div = 0;
	scalar dldiv[nvars];
	for(int k = 0; k < nvars; k++)
		dldiv[k] = 0;

	for(int j = 0; j < NDIM; j++)
	{
		div += grad[j][j+1];
		for(int k = 0; k < nvars; k++)
			dldiv[k] += dgrad[j][j+1][k];
	}

	const scalar ldiv = 2.0/3.0*mu*div;
	for(int k = 0; k < nvars; k++)
		dldiv[k] = 2.0/3.0 * (dmu[k]*div + mu*dldiv[k]);

	for(int i = 0; i < NDIM; i++)
	{
		for(int j = 0; j < NDIM; j++)
		{
			stress[i][j] = mu*(grad[i][j+1] + grad[j][i+1]);

			for(int k = 0; k < nvars; k++)
				dstress[i][j][k]= dmu[k]*(grad[i][j+1] + grad[j][i+1])
				                  + mu*(dgrad[i][j+1][k] + dgrad[j][i+1][k]);
		}

		stress[i][i] -= ldiv;
		for(int k = 0; k < nvars; k++)
			dstress[i][i][k] -= dldiv[k];
	}
}

template class Physics<freal>;
template class IdealGasPhysics<freal>;

#ifdef USE_ADOLC
template class Physics<adouble>;
template class IdealGasPhysics<adouble>;
#endif

}

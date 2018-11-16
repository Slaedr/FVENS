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
IdealGasPhysics<scalar>::IdealGasPhysics(const a_real _g, const a_real M_inf,
		const a_real T_inf, const a_real Re_inf, const a_real _Pr)
	: g{_g}, Minf{M_inf}, Tinf{T_inf}, Reinf{Re_inf}, Pr{_Pr}, sC{110.5}
{
#ifdef DEBUG
	std::cout << " IdealGasPhysics: Physical parameters:\n";
	std::cout << "  Adiabatic index = " <<g << ", M_infty = " <<Minf << ", T_infty = " << Tinf
		<< "\n   Re_infty = " << Reinf << ", Pr = " << Pr << std::endl;
#endif
}

template <typename scalar>
void IdealGasPhysics<scalar>::getDirectionalFluxFromConserved(const scalar *const u, const scalar* const n,
		scalar *const __restrict flux) const
{
	const scalar vn = dimDotProduct(&u[1],n)/u[0];
	const scalar p = getPressure(u[NVARS-1] - 0.5*dimDotProduct(&u[1],&u[1])/u[0]);
	getDirectionalFlux(u, n, vn, p, flux);
}

/** The reference density and reference velocity are the free-stream values, therefore
 * the non-dimensional free-stream density and velocity magnitude are 1.0.
 */
template <typename scalar>
std::array<a_real,NVARS> IdealGasPhysics<scalar>::compute_freestream_state(const a_real aoa) const
{
	std::array<a_real,NVARS> uinf;
	uinf[0] = 1.0;
	uinf[1] = cos(aoa);
	uinf[2] = sin(aoa);
    uinf[3] = getvalue<scalar>(getEnergyFromPressure(getFreestreamPressure(),1.0,1.0));
	return uinf;
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianDirectionalFluxWrtConserved(const scalar *const u,
		const scalar* const n,
		scalar *const __restrict dfdu) const
{
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
}

template <typename scalar>
void IdealGasPhysics<scalar>
::getJacobianVarsWrtConserved(const scalar *const uc, const scalar *const n,
                              scalar *const __restrict dv, scalar *const __restrict dvn,
                              scalar *const __restrict dp, scalar *const __restrict dH) const
{
	for(int j = 0; j < NDIM; j++)
	{
		dv[j*NVARS+0] += -uc[j+1]/(uc[0]*uc[0]);
		dv[j*NVARS+j+1] += 1.0/uc[0];
	}

	for(int j = 0; j < NDIM; j++) {
		dvn[0] += dv[j*NVARS]*n[j];
		dvn[j+1] += n[j]/uc[0];
	}

	const scalar p = getPressureFromConserved(uc);
	getJacobianPressureWrtConserved(uc, dp);

	dH[0] += (dp[0]*uc[0] - (uc[3]+p))/(uc[0]*uc[0]);
	for(int j = 1; j < NDIM+1; j++)
		dH[j] += dp[j]/uc[0];
	dH[3] += (1.0+dp[3])/uc[0];
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianPrimitive2WrtConserved(const scalar *const uc,
		scalar *const __restrict jac) const
{
	jac[0] += 1.0;

	const scalar rho2vmag2 = dimDotProduct(&uc[1],&uc[1]);

	for(int idim = 1; idim < NDIM+1; idim++) {
		// d(up[idim])
		jac[idim*NVARS+0] += -uc[idim]/(uc[0]*uc[0]);
		jac[idim*NVARS+idim] += 1.0/uc[0];
	}

	const scalar p = getPressure(uc[NVARS-1] - 0.5*rho2vmag2/uc[0]);
	scalar dp[NVARS];
	zeros(dp, NVARS);
	getJacobianPressureWrtConserved(uc, rho2vmag2, dp);

	getJacobianTemperature(uc[0], p, dp, &jac[3*NVARS]);
}

template <typename scalar>
void IdealGasPhysics<scalar>::getJacobianStress(const scalar mu, const scalar *const dmu,
		const scalar grad[NDIM][NVARS], const scalar dgrad[NDIM][NVARS][NVARS],
		scalar stress[NDIM][NDIM],
		scalar dstress[NDIM][NDIM][NVARS]) const
{
	scalar div = 0;
	scalar dldiv[NVARS];
	for(int k = 0; k < NVARS; k++)
		dldiv[k] = 0;

	for(int j = 0; j < NDIM; j++)
	{
		div += grad[j][j+1];
		for(int k = 0; k < NVARS; k++)
			dldiv[k] += dgrad[j][j+1][k];
	}

	const scalar ldiv = 2.0/3.0*mu*div;
	for(int k = 0; k < NVARS; k++)
		dldiv[k] = 2.0/3.0 * (dmu[k]*div + mu*dldiv[k]);

	for(int i = 0; i < NDIM; i++)
	{
		for(int j = 0; j < NDIM; j++)
		{
			stress[i][j] = mu*(grad[i][j+1] + grad[j][i+1]);

			for(int k = 0; k < NVARS; k++)
				dstress[i][j][k]= dmu[k]*(grad[i][j+1] + grad[j][i+1])
				                  + mu*(dgrad[i][j+1][k] + dgrad[j][i+1][k]);
		}

		stress[i][i] -= ldiv;
		for(int k = 0; k < NVARS; k++)
			dstress[i][i][k] -= dldiv[k];
	}
}

template class Physics<a_real>;
template class IdealGasPhysics<a_real>;

#ifdef USE_ADOLC
template class Physics<adouble>;
template class IdealGasPhysics<adouble>;
#endif

}

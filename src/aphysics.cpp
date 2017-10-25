/** \file aphysics.cpp
 * \brief Implementation of analytical flux computation and variable conversions
 * \author Aditya Kashi
 * \date 2017 September 13
 */

#include "aphysics.hpp"
	
namespace acfd {

void IdealGasPhysics::getDirectionalFluxFromConserved(const a_real *const u, const a_real* const n, 
		a_real *const __restrict flux) const
{
	const a_real vn = dimDotProduct(&u[1],n)/u[0];
	const a_real p = (g-1.0)*(u[NVARS-1] - 0.5*dimDotProduct(&u[1],&u[1])/u[0]);
	getDirectionalFlux(u, n, vn, p, flux);
}

const std::array<a_real,NVARS> IdealGasPhysics::compute_freestream_state(const a_real aoa) const
{
	std::array<a_real,NVARS> uinf;
	// note that reference density and reference velocity are the values at infinity
	uinf[0] = 1.0;
	uinf[1] = cos(aoa);
	uinf[2] = sin(aoa);
	uinf[3] = getEnergyFromPressure(getFreestreamPressure(),1.0,1.0);
	return uinf;
}

void IdealGasPhysics::getJacobianDirectionalFluxWrtConserved(const a_real *const u, 
		const a_real* const n, 
		a_real *const __restrict dfdu) const
{
	const a_real rhovn = dimDotProduct(&u[1],n), u02 = u[0]*u[0];
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

void IdealGasPhysics::getJacobianVarsWrtConserved(const a_real *const uc, const a_real *const n,
	a_real *const __restrict dvx, a_real *const __restrict dvy, a_real *const __restrict dvn,
	a_real *const __restrict dp, a_real *const __restrict dH) const
{
	dvx[0] += -uc[1]/(uc[0]*uc[0]);
	dvx[1] += 1.0/uc[0];

	dvy[0] += -uc[2]/(uc[0]*uc[0]);
	dvy[2] += 1.0/uc[0];

	dvn[0] += dvx[0]*n[0]+dvy[0]*n[1];
	dvn[1] += n[0]/uc[0];
	dvn[2] += n[1]/uc[0];

	const a_real p = getPressureFromConserved(uc);
	getJacobianPressureWrtConserved(uc, dp);

	dH[0] += (dp[0]*uc[0] - (uc[3]+p))/(uc[0]*uc[0]);
	dH[1] += dp[1]/uc[0];
	dH[2] += dp[2]/uc[0];
	dH[3] += (1.0+dp[3])/uc[0];
}

void IdealGasPhysics::getJacobianPrimitive2WrtConserved(const a_real *const uc, 
		a_real *const __restrict jac) const
{
	jac[0] += 1.0;

	const a_real rho2vmag2 = dimDotProduct(&uc[1],&uc[1]);
	
	for(int idim = 1; idim < NDIM+1; idim++) {
		// d(up[idim])
		jac[idim*NVARS+0] += -uc[idim]/(uc[0]*uc[0]);
		jac[idim*NVARS+idim] += 1.0/uc[0];
	}
	
	const a_real p = getPressure(uc[NVARS-1] - 0.5*rho2vmag2/uc[0]);
	a_real dp[NVARS];
	getJacobianPressureWrtConserved(uc, rho2vmag2, dp);

	getJacobianTemperature(uc[0], p, dp, &jac[3*NVARS]);
}

void IdealGasPhysics::getJacobianStress(const a_real mu, const a_real *const dmu,
		const a_real grad[NDIM][NVARS], const a_real dgrad[NDIM][NVARS][NVARS],
		a_real stress[NDIM][NDIM], 
		a_real dstress[NDIM][NDIM][NVARS]) const
{
	a_real div = 0;
	a_real dldiv[NVARS];
	for(int k = 0; k < NVARS; k++)
		dldiv[k] = 0;

	for(int j = 0; j < NDIM; j++) 
	{
		div += grad[j][j+1];
		for(int k = 0; k < NVARS; k++)
			dldiv[k] += dgrad[j][j+1][k];
	}
	
	const a_real ldiv = 2.0/3.0*mu*div;
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

}

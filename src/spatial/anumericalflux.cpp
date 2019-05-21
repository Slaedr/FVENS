/** \file anumericalflux.cpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015, June 2017, September-October 2017
 */

/* Tapenade notes:
 * No consts
 * #defines work but get substituted
 */
#include <iostream>
#include "anumericalflux.hpp"
#include "physics/aphysics_defs.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template <typename scalar, typename j_real>
InviscidFlux<scalar,j_real>::InviscidFlux(const IdealGasPhysics<scalar> *const phyctx)
	: physics(phyctx), jphy {
	new IdealGasPhysics<j_real>(phyctx->g, phyctx->Minf, phyctx->Tinf, phyctx->Reinf, phyctx->Pr)},
	  g{phyctx->g}
{ }

template <typename scalar, typename j_real>
InviscidFlux<scalar,j_real>::~InviscidFlux()
{
	delete jphy;
}

template <typename scalar, typename j_real>
LocalLaxFriedrichsFlux<scalar,j_real>
::LocalLaxFriedrichsFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: InviscidFlux<scalar>(analyticalflux)
{ }

template <typename scalar, typename j_real>
void LocalLaxFriedrichsFlux<scalar,j_real>::get_flux(const scalar *const ul,
                                                     const scalar *const ur,
                                                     const scalar *const n,
                                                     scalar *const __restrict flux) const
{
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0],pi);
	cj = physics->getSoundSpeed(ur[0],pj);

	const scalar eig =
		std::fabs(vni)+ci > std::fabs(vnj)+cj ? std::fabs(vni)+ci : std::fabs(vnj)+cj;

	physics->getDirectionalFluxFromConserved(ul,n,flux);
	scalar fluxr[NVARS];
	physics->getDirectionalFluxFromConserved(ur,n,fluxr);
	for(int i = 0; i < NVARS; i++) {
		flux[i] = 0.5*( flux[i] + fluxr[i] - eig*(ur[i]-ul[i]) );
	}
}

/** Jacobian with frozen spectral radius
 */
template <typename scalar, typename j_real>
void LocalLaxFriedrichsFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
                                                         const j_real* const n,
                                                         j_real *const __restrict dfdl,
                                                         j_real *const __restrict dfdr) const
{
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj, eig;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0],pi);
	cj = physics->getSoundSpeed(ur[0],pj);

	// max eigenvalue
	if( fabs(vni)+ci >= fabs(vnj)+cj )
	{
		eig = fabs(vni)+ci;
	}
	else
	{
		eig = fabs(vnj)+cj;
	}

	// get flux jacobians
	physics->getJacobianDirectionalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianDirectionalFluxWrtConserved(ur, n, dfdr);

	// add contributions to left derivative
	for(int i = 0; i < NVARS; i++)
		dfdl[i*NVARS+i] -= -eig;

	// add contributions to right derivarive
	for(int i = 0; i < NVARS; i++)
		dfdr[i*NVARS+i] -= eig;

	for(int i = 0; i < NVARS; i++)
		for(int j = 0; j < NVARS; j++)
		{
			// lower block
			dfdl[i*NVARS+j] = -0.5*dfdl[i*NVARS+j];
			// upper block
			dfdr[i*NVARS+j] =  0.5*dfdr[i*NVARS+j];
		}
}

// full linearization; no better than frozen version
template <typename scalar, typename j_real>
void LocalLaxFriedrichsFlux<scalar,j_real>::get_jacobian_2(const j_real *const ul,
		const j_real *const ur,
		const j_real* const n,
		j_real *const __restrict dfdl, j_real *const __restrict dfdr) const
{
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj, eig;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0],pi);
	cj = physics->getSoundSpeed(ur[0],pj);

	// max eigenvalue
	bool leftismax;
	if( fabs(vni)+ci >= fabs(vnj)+cj )
	{
		eig = fabs(vni)+ci;
		leftismax = true;
	}
	else
	{
		eig = fabs(vnj)+cj;
		leftismax = false;
	}

	// get flux jacobians
	physics->getJacobianDirectionalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianDirectionalFluxWrtConserved(ur, n, dfdr);

	// linearization of the dissipation term

	const j_real ctermi = 0.5 /
		sqrt( g*(g-1)/ul[0]* (ul[3]-(ul[1]*ul[1]+ul[2]*ul[2])/(2*ul[0])) );
	const j_real ctermj = 0.5 /
		sqrt( g*(g-1)/ur[0]* (ur[3]-(ur[1]*ur[1]+ur[2]*ur[2])/(2*ur[0])) );
	j_real dedu[NVARS];

	if(leftismax) {
		dedu[0] = -fabs(vni/(ul[0]*ul[0])) + ctermi*g*(g-1)*( -ul[3]/(ul[0]*ul[0])
				+ (ul[1]*ul[1]+ul[2]*ul[2])/(ul[0]*ul[0]*ul[0]) );
		dedu[1] = (vni>0 ? n[0]/ul[0] : -n[0]/ul[0]) + ctermi*g*(g-1)*(-ul[1]/ul[0]);
		dedu[2] = (vni>0 ? n[1]/ul[0] : -n[1]/ul[0]) + ctermi*g*(g-1)*(-ul[2]/ul[0]);
		dedu[3] = ctermi*g*(g-1)/ul[0];
	}
	else {
		for(int i = 0; i < NVARS; i++)
			dedu[i] = 0;
	}

	// add contributions to left derivative
	for(int i = 0; i < NVARS; i++)
	{
		dfdl[i*NVARS+i] -= -eig;
		for(int j = 0; j < NVARS; j++)
			dfdl[i*NVARS+j] -= dedu[j]*(ur[i]-ul[i]);
	}

	if(leftismax) {
		for(int i = 0; i < NVARS; i++)
			dedu[i] = 0;
	} else {
		dedu[0] = -fabs(vnj/(ur[0]*ur[0])) + ctermj*g*(g-1)*( -ur[3]/(ur[0]*ur[0])
				+ (ur[1]*ur[1]+ur[2]*ur[2])/(ur[0]*ur[0]*ur[0]) );
		dedu[1] = (vnj>0 ? n[0]/ur[0] : -n[0]/ur[0]) + ctermj*g*(g-1)*(-ur[1]/ur[0]);
		dedu[2] = (vnj>0 ? n[1]/ur[0] : -n[1]/ur[0]) + ctermj*g*(g-1)*(-ur[2]/ur[0]);
		dedu[3] = ctermj*g*(g-1)/ur[0];
	}

	// add contributions to right derivarive
	for(int i = 0; i < NVARS; i++)
	{
		dfdr[i*NVARS+i] -= eig;
		for(int j = 0; j < NVARS; j++)
			dfdr[i*NVARS+j] -= dedu[j]*(ur[i]-ul[i]);
	}

	for(int i = 0; i < NVARS; i++)
		for(int j = 0; j < NVARS; j++)
		{
			// lower block
			dfdl[i*NVARS+j] = -0.5*dfdl[i*NVARS+j];
			// upper block
			dfdr[i*NVARS+j] =  0.5*dfdr[i*NVARS+j];
		}
}

template <typename scalar, typename j_real>
VanLeerFlux<scalar,j_real>::VanLeerFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: InviscidFlux<scalar>(analyticalflux)
{
}

template <typename scalar, typename j_real>
void VanLeerFlux<scalar,j_real>::get_flux(const scalar *const ul, const scalar *const ur,
		const scalar* const n, scalar *const __restrict flux) const
{
	scalar fiplus[NVARS], fjminus[NVARS];

	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0],pi);
	cj = physics->getSoundSpeed(ur[0],pj);

	//Normal mach numbers
	const scalar Mni = vni/ci;
	const scalar Mnj = vnj/cj;

	//Calculate split fluxes
	if(Mni < -1.0)
		for(int i = 0; i < NVARS; i++)
			fiplus[i] = 0;
	else if(Mni > 1.0)
		physics->getDirectionalFlux(ul,n,vni,pi,fiplus);
	else
	{
		const scalar vmags = pow(ul[1]/ul[0], 2) + pow(ul[2]/ul[0], 2);
		fiplus[0] = ul[0]*ci*pow(Mni+1, 2)/4.0;
		fiplus[1] = fiplus[0] * (ul[1]/ul[0] + n[0]*(2.0*ci - vni)/g);
		fiplus[2] = fiplus[0] * (ul[2]/ul[0] + n[1]*(2.0*ci - vni)/g);
		fiplus[3] = fiplus[0] * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
	}

	if(Mnj > 1.0)
		for(int i = 0; i < NVARS; i++)
			fjminus[i] = 0;
	else if(Mnj < -1.0)
		physics->getDirectionalFlux(ur,n,vnj,pj,fjminus);
	else
	{
		const scalar vmags = pow(ur[1]/ur[0], 2) + pow(ur[2]/ur[0], 2);
		fjminus[0] = -ur[0]*cj*pow(Mnj-1, 2)/4.0;
		fjminus[1] = fjminus[0] * (ur[1]/ur[0] + n[0]*(-2.0*cj - vnj)/g);
		fjminus[2] = fjminus[0] * (ur[2]/ur[0] + n[1]*(-2.0*cj - vnj)/g);
		fjminus[3] = fjminus[0] * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
	}

	//Update the flux vector
	for(int i = 0; i < NVARS; i++)
		flux[i] = fiplus[i] + fjminus[i];
}

template <typename scalar, typename j_real>
void VanLeerFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
		const j_real* const n, j_real *const dfdl, j_real *const dfdr) const
{
	throw " ! VanLeerFlux: Not implemented!";
}

template <typename scalar, typename j_real>
AUSMFlux<scalar,j_real>::AUSMFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: InviscidFlux<scalar>(analyticalflux)
{ }

template <typename scalar, typename j_real>
void AUSMFlux<scalar,j_real>::get_flux(const scalar *const ul, const scalar *const ur,
                                       const scalar* const n, scalar *const __restrict flux) const
{
	scalar ML, MR, pL, pR;
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	const scalar ci = physics->getSoundSpeed(ul[0],pi);
	const scalar cj = physics->getSoundSpeed(ur[0],pj);

	const scalar Mni = vni/ci, Mnj = vnj/cj;

	// split non-dimensional convection speeds (ie split Mach numbers) and split pressures
	if(fabs(Mni) <= 1.0)
	{
		ML = 0.25*(Mni+1)*(Mni+1);
		pL = ML*pi*(2.0-Mni);
	}
	else if(Mni < -1.0) {
		ML = 0;
		pL = 0;
	}
	else {
		ML = Mni;
		pL = pi;
	}

	if(fabs(Mnj) <= 1.0) {
		MR = -0.25*(Mnj-1)*(Mnj-1);
		pR = -MR*pj*(2.0+Mnj);
	}
	else if(Mnj < -1.0) {
		MR = Mnj;
		pR = pj;
	}
	else {
		MR = 0;
		pR = 0;
	}

	// Interface convection speed and pressure
	const scalar Mhalf = ML+MR;
	const scalar phalf = pL+pR;

	// Fluxes
	flux[0] = Mhalf/2.0*(ul[0]*ci+ur[0]*cj) -fabs(Mhalf)/2.0*(ur[0]*cj-ul[0]*ci);
	for(int j = 1; j < NDIM+1; j++)
		flux[j] = Mhalf/2.0*(ul[j]*ci+ur[j]*cj) -fabs(Mhalf)/2.0*(ur[j]*cj-ul[j]*ci) + phalf*n[j-1];
	flux[3] = Mhalf/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj))
		-fabs(Mhalf)/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi));
}

template <typename scalar, typename j_real>
void AUSMFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
                                           const j_real* const n,
                                           j_real *const dfdl, j_real *const dfdr) const
{
	j_real ML, MR;
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	const j_real ci = physics->getSoundSpeed(ul[0],pi);
	const j_real cj = physics->getSoundSpeed(ur[0],pj);

	const j_real Mni = vni/ci, Mnj = vnj/cj;

	j_real dpi[NVARS], dci[NVARS], dpj[NVARS], dcj[NVARS], dmni[NVARS], dmnj[NVARS];
	j_real dML[NVARS], dMR[NVARS], dpL[NVARS], dpR[NVARS];
	for(int i = 0; i < NVARS; i++) {
		dpi[i] = 0; dci[i] = 0; dpj[i] = 0; dcj[i] = 0; dmni[i] = 0; dmnj[i] = 0;
		dML[i] = dMR[i] = dpL[i] = dpR[i] = 0;
	}
	physics->getJacobianPressureWrtConserved(ul, dpi);
	physics->getJacobianPressureWrtConserved(ur, dpj);
	physics->getJacobianSoundSpeed(ul[0], pi, dpi, ci, dci);
	physics->getJacobianSoundSpeed(ur[0], pj, dpj, cj, dcj);

	dmni[0] = (-1.0/(ul[0]*ul[0])*(ul[1]*n[0]+ul[2]*n[1])*ci - vni*dci[0])/(ci*ci);
	dmni[1] = (n[0]/ul[0]*ci - vni*dci[1])/(ci*ci);
	dmni[2] = (n[1]/ul[0]*ci - vni*dci[2])/(ci*ci);
	dmni[3] = -vni*dci[3]/(ci*ci);

	dmnj[0] = (-1.0/(ur[0]*ur[0])*(ur[1]*n[0]+ur[2]*n[1])*cj - vnj*dcj[0])/(cj*cj);
	dmnj[1] = (n[0]/ur[0]*cj - vnj*dcj[1])/(cj*cj);
	dmnj[2] = (n[1]/ur[0]*cj - vnj*dcj[2])/(cj*cj);
	dmnj[3] = -vnj*dcj[3]/(cj*cj);

	// split non-dimensional convection speeds (ie split Mach numbers) and split pressures
	if(fabs(Mni) <= 1.0)
	{
		ML = 0.25*(Mni+1)*(Mni+1);
		for(int k = 0; k < NVARS; k++)
			dML[k] = 0.5*(Mni+1)*dmni[k];

		// pL = ML*pi*(2.0-Mni)
		for(int k = 0; k < NVARS; k++)
			dpL[k] = dML[k]*pi*(2.0-Mni) + ML*dpi[k]*(2.0-Mni) - ML*pi*dmni[k];
	}
	else if(Mni < -1.0) {
		ML = 0;
	}
	else {
		ML = Mni;
		// pL = pi
		for(int k = 0; k < NVARS; k++) {
			dML[k] = dmni[k];
			dpL[k] = dpi[k];
		}
	}

	if(fabs(Mnj) <= 1.0)
	{
		MR = -0.25*(Mnj-1)*(Mnj-1);
		for(int k = 0; k < NVARS; k++)
			dMR[k] = -0.5*(Mnj-1)*dmnj[k];

		// pR = -MR*pj*(2.0+Mnj);
		for(int k = 0; k < NVARS; k++)
			dpR[k] = -dMR[k]*pj*(2.0+Mnj) - MR*dpj[k]*(2.0+Mnj) - MR*pj*dmnj[k];
	}
	else if(Mnj < -1.0) {
		MR = Mnj;
		// pR = pj;
		for(int k = 0; k < NVARS; k++) {
			dMR[k] = dmnj[k];
			dpR[k] = dpj[k];
		}
	}
	else {
		MR = 0;
	}

	// Interface convection speed and pressure
	const j_real Mhalf = ML+MR;
	//const j_real phalf = pL+pR;

	/* Note that derivative of Mhalf w.r.t. ul is dML and that w.r.t. ur dMR,
	 * and similarly for phalf.
	 */

	// mass flux
	//flux[0] = Mhalf/2.0*(ul[0]*ci+ur[0]*cj) -fabs(Mhalf)/2.0*(ur[0]*cj-ul[0]*ci);

	dfdl[0] = dML[0]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*(ci+ul[0]*dci[0])
		-( (Mhalf>=0 ? 1.0 : -1.0)*dML[0]/2.0*(ur[0]*cj-ul[0]*ci)
				+ fabs(Mhalf)/2.0*(-ci-ul[0]*dci[0]) );

	dfdr[0] = dMR[0]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*(cj+ur[0]*dcj[0])
		-( (Mhalf>=0 ? 1.0 : -1.0)*dMR[0]/2.0*(ur[0]*cj-ul[0]*ci)
				+ fabs(Mhalf)/2.0*(cj+ur[0]*dcj[0]) );

	for(int k = 1; k < NVARS; k++)
	{
	  dfdl[k] = dML[k]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*ul[0]*dci[k] -
		( (Mhalf>=0 ? 1.0:-1.0)*dML[k]/2.0*(ur[0]*cj-ul[0]*ci) - fabs(Mhalf)/2.0*ul[0]*dci[k] );
	  dfdr[k] = dMR[k]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*ur[0]*dcj[k] -
		( (Mhalf>=0 ? 1.0:-1.0)*dMR[k]/2.0*(ur[0]*cj-ul[0]*ci) + fabs(Mhalf)/2.0*ur[0]*dcj[k] );
	}

	// momentum
	//flux[j] = Mhalf/2.0*(ul[j]*ci+ur[j]*cj) -fabs(Mhalf)/2.0*(ur[j]*cj-ul[j]*ci) + phalf*n[j-1];

	for(int j = 1; j < NDIM+1; j++)
	{
		dfdl[j*NVARS+j] = dML[j]/2.0*(ul[j]*ci+ur[j]*cj) + Mhalf/2.0*(ci+ul[j]*dci[j]) -
			( (Mhalf>=0? 1.0:-1.0)*dML[j]/2.0*(ur[j]*cj-ul[j]*ci) + fabs(Mhalf)/2.0*(-ci-ul[j]*dci[j]) )
			+ dpL[j]*n[j-1];
		dfdr[j*NVARS+j] = dMR[j]/2.0*(ul[j]*ci+ur[j]*cj) + Mhalf/2.0*(cj+ur[j]*dcj[j]) -
			( (Mhalf>=0? 1.0:-1.0)*dMR[j]/2.0*(ur[j]*cj-ul[j]*ci) + fabs(Mhalf)/2.0*(cj+ur[j]*dcj[j]) )
			+ dpR[j]*n[j-1];
		for(int k = 0; k < NVARS; k++)
		{
			if(k == j) continue;
			dfdl[j*NVARS+k] = dML[k]/2.0*(ul[j]*ci+ur[j]*cj) + Mhalf/2.0*ul[j]*dci[k] -
				( (Mhalf>=0?1.0:-1.0)*dML[k]/2.0*(ur[j]*cj-ul[j]*ci) - fabs(Mhalf)/2.0*ul[j]*dci[k] )
				+ dpL[k]*n[j-1];
			dfdr[j*NVARS+k] = dMR[k]/2.0*(ul[j]*ci+ur[j]*cj) + Mhalf/2.0*ur[j]*dcj[k] -
				( (Mhalf>=0?1.0:-1.0)*dMR[k]/2.0*(ur[j]*cj-ul[j]*ci) + fabs(Mhalf)/2.0*ur[j]*dcj[k] )
				+ dpR[k]*n[j-1];
		}
	}

	// Energy flux
	//flux[3]=Mhalf/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj))-fabs(Mhalf)/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi));

	dfdl[3*NVARS+3] =
		dML[3]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) + Mhalf/2.0*(dci[3]*(ul[3]+pi)+ci*(1.0+dpi[3])) -
		( (Mhalf>=0?1.0:-1.0)*dML[3]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi))
		  +fabs(Mhalf)/2.0*(-dci[3]*(ul[3]+pi)-ci*(1.0+dpi[3])) );
	dfdr[3*NVARS+3] =
		dMR[3]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) + Mhalf/2.0*(dcj[3]*(ur[3]+pj)+cj*(1.0+dpj[3])) -
		( (Mhalf>=0?1.0:-1.0)*dMR[3]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi))
		  +fabs(Mhalf)/2.0*(dcj[3]*(ur[3]+pj)+cj*(1.0+dpj[3])) );
	for(int k = 0; k < NVARS-1; k++)
	{
		dfdl[3*NVARS+k] =
		  dML[k]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) +Mhalf/2.0*(dci[k]*(ul[3]+pi)+ci*dpi[k]) -
		  ( (Mhalf>=0?1.0:-1.0)*dML[k]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi))
			+fabs(Mhalf)/2.0*(-dci[k]*(ul[3]+pi)-ci*dpi[k]) );
		dfdr[3*NVARS+k] =
		  dMR[k]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) +Mhalf/2.0*(dcj[k]*(ur[3]+pj)+cj*dpj[k]) -
		  ( (Mhalf>=0?1.0:-1.0)*dMR[k]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi))
			+fabs(Mhalf)/2.0*(dcj[k]*(ur[3]+pj)+cj*dpj[k]) );
	}

	for(int k = 0; k < NVARS*NVARS; k++)
		dfdl[k] = -dfdl[k];
}

template <typename scalar, typename j_real>
AUSMPlusFlux<scalar,j_real>::AUSMPlusFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: InviscidFlux<scalar>(analyticalflux)
{ }

template <typename scalar, typename j_real>
void AUSMPlusFlux<scalar,j_real>::get_flux(const scalar *const ul, const scalar *const ur,
		const scalar* const n, scalar *const __restrict flux) const
{
	scalar ML, MR, pL, pR;
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	const scalar ci = physics->getSoundSpeed(ul[0],pi);
	const scalar cj = physics->getSoundSpeed(ur[0],pj);
	const scalar vmag2i = dimDotProduct(vi,vi);
	const scalar vmag2j = dimDotProduct(vj,vj);

	// Interface speed of sound
	scalar csi = sqrt((ci*ci/(g-1.0)+0.5*vmag2i)*2.0*(g-1.0)/(g+1.0));
	scalar csj = sqrt((cj*cj/(g-1.0)+0.5*vmag2j)*2.0*(g-1.0)/(g+1.0));
	scalar corri, corrj;
	if(csi > vni)
		corri = csi;
	else
		corri = vni;
	if(csj > -vnj)
		corrj = csj;
	else
		corrj = -vnj;
	csi = csi*csi/corri;
	csj = csj*csj/corrj;
	const scalar chalf = (csi < csj) ? csi : csj;

	const scalar Mni = vni/chalf, Mnj = vnj/chalf;

	// split non-dimensional convection speeds (ie split Mach numbers) and split pressures
	if(fabs(Mni) <= 1.0)
	{
		ML = 0.25*(Mni+1)*(Mni+1) + 1.0/8.0*(Mni*Mni-1.0)*(Mni*Mni-1.0);
		pL = pi*(0.25*(Mni+1)*(Mni+1)*(2.0-Mni) + 3.0/16*Mni*(Mni*Mni-1.0)*(Mni*Mni-1.0));
	}
	else if(Mni < -1.0) {
		ML = 0;
		pL = 0;
	}
	else {
		ML = Mni;
		pL = pi;
	}

	if(fabs(Mnj) <= 1.0) {
		MR = -0.25*(Mnj-1)*(Mnj-1) - 1.0/8.0*(Mnj*Mnj-1.0)*(Mnj*Mnj-1.0);
		pR = pj*(0.25*(Mnj-1)*(Mnj-1)*(2.0+Mnj) - 3.0/16*Mnj*(Mnj*Mnj-1.0)*(Mnj*Mnj-1.0));
	}
	else if(Mnj < -1.0) {
		MR = Mnj;
		pR = pj;
	}
	else {
		MR = 0;
		pR = 0;
	}

	// Interface convection speed and pressure
	const scalar Mhalf = ML+MR;
	const scalar phalf = pL+pR;

	// Fluxes
	flux[0] = chalf* (Mhalf/2.0*(ul[0]+ur[0]) -fabs(Mhalf)/2.0*(ur[0]-ul[0]));
	for(int j = 1; j < NDIM+1; j++)
		flux[j] = chalf* (Mhalf/2.0*(ul[j]+ur[j]) -fabs(Mhalf)/2.0*(ur[j]-ul[j])) + phalf*n[j-1];
	flux[3] = chalf* (Mhalf/2.0*(ul[3]+pi+ur[3]+pj) -fabs(Mhalf)/2.0*((ur[3]+pj)-(ul[3]+pi)));

	/*const scalar mplus = 0.5*(Mhalf + std::fabs(Mhalf)), mminus = 0.5*(Mhalf - std::fabs(Mhalf));
	flux[0] =  (mplus*ci*ul[0] + mminus*cj*ur[0]);
	flux[1] =  (mplus*ci*ul[1] + mminus*cj*ur[1]) + phalf*n[0];
	flux[2] =  (mplus*ci*ul[2] + mminus*cj*ur[2]) + phalf*n[1];
	flux[3] =  (mplus*ci*(ul[3]+pi) + mminus*cj*(ur[3]+pj));*/
}

template <typename scalar, typename j_real>
void AUSMPlusFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
		const j_real* const n, j_real *const dfdl, j_real *const dfdr) const
{
	std::cout << " ! AUSMPlusFlux: Not implemented!\n";
}

template <typename scalar, typename j_real>
RoeAverageBasedFlux<scalar,j_real>::RoeAverageBasedFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: InviscidFlux<scalar>(analyticalflux)
{ }

template <typename scalar, typename j_real>
inline
void RoeAverageBasedFlux<scalar,j_real>::getJacobiansRoeAveragesWrtConserved (
	const j_real ul[NVARS], const j_real ur[NVARS], const j_real n[NDIM],
	const j_real vxi, const j_real vyi, const j_real Hi,
	const j_real vxj, const j_real vyj, const j_real Hj,
	const j_real dvxi[NVARS], const j_real dvyi[NVARS], const j_real dHi[NVARS],
	const j_real dvxj[NVARS], const j_real dvyj[NVARS], const j_real dHj[NVARS],
	j_real dRiji[NVARS], j_real drhoiji[NVARS], j_real dvxiji[NVARS], j_real dvyiji[NVARS],
	j_real dvm2iji[NVARS], j_real dvniji[NVARS], j_real dHiji[NVARS], j_real dciji[NVARS],
	j_real dRijj[NVARS], j_real drhoijj[NVARS], j_real dvxijj[NVARS], j_real dvyijj[NVARS],
	j_real dvm2ijj[NVARS], j_real dvnijj[NVARS], j_real dHijj[NVARS], j_real dcijj[NVARS] ) const
{
	j_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;

	j_real vi[NDIM], vj[NDIM], vij[NDIM];
	vi[0] = vxi; vi[1] = vyi; vj[0] = vxj; vj[1] = vyj;

	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);

	vxij = vij[0]; vyij = vij[1];

	// Derivatives of Rij
	dRiji[0] = 0.5/Rij * (-ur[0])/(ul[0]*ul[0]);
	dRijj[0] = 0.5/Rij / ul[0];
	for(int k = 1; k < NVARS; k++)
	{
		dRiji[k] = 0;
		dRijj[k] = 0;
	}

	const j_real rden2 = (Rij+1.0)*(Rij+1.0);

	// derivatives of Roe velocities
	// Note: vxij = (Rij * ur[1]/ur[0] + ul[1]/ul[0]) / (Rij+1)
	dvxiji[0] = ((dRiji[0]*ur[1]/ur[0] -ul[1]/(ul[0]*ul[0]))*(Rij+1.0)
			-(Rij*vxj+vxi)*dRiji[0])/rden2;
	dvxiji[1] = ((dRiji[1]*ur[1]/ur[0] + 1.0/ul[0])*(Rij+1.0)-(Rij*vxj+vxi)*dRiji[1])/rden2;
	dvxiji[2] = (dRiji[2]*ur[1]/ur[0] *(Rij+1.0)- (Rij*vxj+vxi)*dRiji[2])/rden2;
	dvxiji[3] = (dRiji[3]*ur[1]/ur[0] *(Rij+1.0)- (Rij*vxj+vxi)*dRiji[3])/rden2;

	dvxijj[0] = ((dRijj[0]*ur[1]/ur[0] +Rij/(ur[0]*ur[0])*(-ur[1]))*(Rij+1.0)
			-(Rij*vxj+vxi)*dRijj[0]) / rden2;
	dvxijj[1] = ((dRijj[1]*ur[1]/ur[0] +Rij/ur[0])*(Rij+1.0)-(Rij*vxj+vxi)*dRijj[1]) / rden2;
	dvxijj[2] = (dRijj[2]*ur[1]/ur[0] *(Rij+1.0) - (Rij*vxj+vxi)*dRijj[2]) / rden2;
	dvxijj[3] = (dRijj[3]*ur[1]/ur[0] *(Rij+1.0) - (Rij*vxj+vxi)*dRijj[3]) / rden2;

	// Note: vyij = (Rij *ur[2]/ur[0] + ul[2]/ul[0] ) / (Rij+1)
	dvyiji[0] = ((ur[2]/ur[0]*dRiji[0] - ul[2]/(ul[0]*ul[0]))*(Rij+1.0)
			-(Rij*vyj+vyi)*dRiji[0]) / rden2;
	dvyiji[1] = (ur[2]/ur[0]*dRiji[1] *(Rij+1.0) - (Rij*vyj+vyi)*dRiji[1]) / rden2;
	dvyiji[2] = ((ur[2]/ur[0]*dRiji[2] + 1.0/ul[0])*(Rij+1.0) -(Rij*vyj+vyi)*dRiji[2]) / rden2;
	dvyiji[3] = (ur[2]/ur[0]*dRiji[3] *(Rij+1.0) -(Rij*vyj+vyi)*dRiji[3]) / rden2;

	dvyijj[0] = ((dRijj[0]*ur[2]/ur[0] + Rij/(ur[0]*ur[0])*(-ur[2]))*(Rij+1.0)
		-(Rij*vyj+vyi)*dRijj[0] ) / rden2;
	dvyijj[1] = (dRijj[1]*ur[2]/ur[0] *(Rij+1.0) -(Rij*vyj+vyi)*dRijj[1]) / rden2;
	dvyijj[2] = ((dRijj[2]*ur[2]/ur[0] + Rij/ur[0])*(Rij+1.0) -(Rij*vyj+vyi)*dRijj[2]) / rden2;
	dvyijj[3] = (dRijj[3]*ur[2]/ur[0] *(Rij+1.0) - (Rij*vyj+vyi)*dRijj[3]) / rden2;

	// derivative of Roe normal velocity and Roe velocity magnitude
	for(int k = 0; k < NVARS; k++) {
		dvniji[k] = dvxiji[k]*n[0] + dvyiji[k]*n[1];
		dvnijj[k] = dvxijj[k]*n[0] + dvyijj[k]*n[1];
		dvm2iji[k] = 2.0*( vxij*dvxiji[k] + vyij*dvyiji[k] );
		dvm2ijj[k] = 2.0*( vxij*dvxijj[k] + vyij*dvyijj[k] );
	}

	// derivative of Roe speed of sound
	// cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) ) = sqrt( (g-1)*((Rij*Hj+Hi)/(Rij+1) - vm2ij*0.5) )
	for(int k = 0; k < NVARS; k++) {
		dciji[k] = 0.5/cij*(g-1.0)
		  * (((dRiji[k]*Hj+dHi[k])*(Rij+1)-(Rij*Hj+Hi)*dRiji[k])/rden2 - 0.5*dvm2iji[k]);
		dcijj[k] = 0.5/cij*(g-1.0)
		  * (((dRijj[k]*Hj+Rij*dHj[k])*(Rij+1) - (Rij*Hj+Hi)*dRijj[k])/rden2 - 0.5*dvm2ijj[k] );
	}

	// derivatives of Roe-averaged density
	drhoiji[0] = dRiji[0]*ul[0] + Rij;
	drhoijj[0] = dRijj[0]*ul[0];
	for(int k = 1; k < NVARS; k++)
	{
		drhoiji[k] = 0;
		drhoijj[k] = 0;
	}

	// derivatives of Roe-averaged specific enthalpy
	// Hij = (Rij*Hj + Hi)/(Rij + 1.0)
	for(int k = 0; k < NVARS; k++)
	{
		dHiji[k] = ((dRiji[k]*Hj+dHi[k])*(Rij+1.0)-(Rij*Hj+Hi)*dRiji[k])/rden2;
		dHijj[k] = ((dRijj[k]*Hj+Rij*dHj[k])*(Rij+1.0)-(Rij*Hj+Hi)*dRijj[k])/rden2;
	}
}

template <typename scalar, typename j_real>
RoeFlux<scalar,j_real>::RoeFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: RoeAverageBasedFlux<scalar,j_real>(analyticalflux), fixeps{1.0e-4}
{ }

template <typename scalar, typename j_real>
void RoeFlux<scalar,j_real>::get_flux(const scalar *const ul, const scalar *const ur,
                                      const scalar *const n, scalar *const __restrict flux) const
{
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);

	// compute Roe-averages
	scalar Rij,rhoij,vij[NDIM],vm2ij,vnij,Hij,cij;
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);

	// eigenvalues
	scalar l[NVARS];
	l[0] = fabs(vnij-cij);
	for(int j = 1; j < NDIM+1; j++)
		l[j] = fabs(vnij);
	l[NDIM+1] = fabs(vnij+cij);

	// Harten entropy fix
	const scalar delta = fixeps*cij;
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		if(l[ivar] < delta)
			l[ivar] = (l[ivar]*l[ivar] + delta*delta)/(2.0*delta);
	}

	//> A_Roe * dU

	const scalar devn = vnj-vni, dep = pj-pi, derho = ur[0]-ul[0];
	scalar adu[NVARS];

	// product of eigenvalues and wave strengths
	scalar lalpha[NVARS];
	lalpha[0] = l[0]*(dep-rhoij*cij*devn)/(2.0*cij*cij);
	lalpha[1] = l[1]*(derho - dep/(cij*cij));
	lalpha[2] = l[1]*rhoij;
	lalpha[3] = l[3]*(dep+rhoij*cij*devn)/(2.0*cij*cij);

	// un-c:
	adu[0] = lalpha[0];
	adu[1] = lalpha[0]*(vij[0]-cij*n[0]);
	adu[2] = lalpha[0]*(vij[1]-cij*n[1]);
	adu[3] = lalpha[0]*(Hij-cij*vnij);

	// un:
	adu[0] += lalpha[1];
	adu[1] += lalpha[1]*vij[0] +      lalpha[2]*(vj[0]-vi[0] - devn*n[0]);
	adu[2] += lalpha[1]*vij[1] +      lalpha[2]*(vj[1]-vi[1] - devn*n[1]);
	adu[3] += lalpha[1]*vm2ij/2.0 + lalpha[2] *(vij[0]*(vj[0]-vi[0]) +vij[1]*(vj[1]-vi[1]) -vnij*devn);

	// un+c:
	adu[0] += lalpha[3];
	adu[1] += lalpha[3]*(vij[0]+cij*n[0]);
	adu[2] += lalpha[3]*(vij[1]+cij*n[1]);
	adu[3] += lalpha[3]*(Hij+cij*vnij);

	// get one-sided flux vectors
	scalar fi[NVARS], fj[NVARS];
	physics->getDirectionalFlux(ul,n,vni,pi,fi);
	physics->getDirectionalFlux(ur,n,vnj,pj,fj);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
		flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - adu[ivar]);
}

/** \todo Works, but check correctness.
 */
template <typename scalar, typename j_real>
void RoeFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
                                          const j_real* const n,
                                          j_real *const __restrict dfdl,
                                          j_real *const __restrict dfdr) const
{
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);

	const j_real vxi = vi[0], vxj=vj[0], vyi=vi[1], vyj=vj[1];

	// compute Roe-averages
	j_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij,vij[NDIM];
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);
	vxij = vij[0]; vyij = vij[1];

	//> Derivatives of the above variables

	j_real dpi[NVARS], dpj[NVARS], dvni[NVARS], dvnj[NVARS], dvi[NDIM*NVARS], dvj[NDIM*NVARS],
	dHi[NVARS], dHj[NVARS],
	dRiji[NVARS], dRijj[NVARS], dvxiji[NVARS], dvyiji[NVARS], dvxijj[NVARS], dvyijj[NVARS],
	dvniji[NVARS], dvnijj[NVARS],dvm2iji[NVARS], dvm2ijj[NVARS], dciji[NVARS], dcijj[NVARS],
	drhoiji[NVARS], drhoijj[NVARS], dHiji[NVARS], dHijj[NVARS];
	for(int k = 0; k < NVARS; k++)
	{
		dpi[k] = dpj[k] = dHi[k] = dHj[k] = 0;
		dvni[k] = dvnj[k] = 0;
		for(int j = 0; j < NDIM; j++)
		{
			dvi[j*NVARS+k] = 0;
			dvj[j*NVARS+k] = 0;
		}
	}

	physics->getJacobianVarsWrtConserved(ul,n,dvi,dvni,dpi,dHi);
	physics->getJacobianVarsWrtConserved(ur,n,dvj,dvnj,dpj,dHj);

	j_real dvxj[NVARS], dvyj[NVARS], dvxi[NVARS], dvyi[NVARS];
	for(int k = 0; k < NVARS; k++) {
		dvxi[k] = dvi[k];       dvxj[k] = dvj[k];
		dvyi[k] = dvi[NVARS+k]; dvyj[k] = dvj[NVARS+k];
	}

	getJacobiansRoeAveragesWrtConserved(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj,dvxi,dvyi,dHi,dvxj,dvyj,dHj,
		dRiji, drhoiji, dvxiji, dvyiji, dvm2iji, dvniji, dHiji, dciji,
		dRijj, drhoijj, dvxijj, dvyijj, dvm2ijj, dvnijj, dHijj, dcijj);

	//> eigenvalues

	j_real l[NVARS];
	l[0] = fabs(vnij-cij); l[1] = fabs(vnij); l[2] = l[1]; l[3] = fabs(vnij+cij);

	j_real dli[NVARS][NVARS], dlj[NVARS][NVARS];
	for(int k = 0; k < NVARS; k++)
	{
		dli[0][k] = (vnij-cij >= 0 ? 1.0:-1.0)*(dvniji[k]-dciji[k]);
		dli[1][k] = (vnij>=0 ? 1.0:-1.0)*dvniji[k];
		dli[2][k] = dli[1][k];
		dli[3][k] = (vnij+cij >= 0 ? 1.0:-1.0)*(dvniji[k]+dciji[k]);

		dlj[0][k] = (vnij-cij >= 0 ? 1.0:-1.0)*(dvnijj[k]-dcijj[k]);
		dlj[1][k] = (vnij>=0 ? 1.0:-1.0)*dvnijj[k];
		dlj[2][k] = dlj[1][k];
		dlj[3][k] = (vnij+cij >= 0 ? 1.0:-1.0)*(dvnijj[k]+dcijj[k]);
	}

	// Harten entropy fix
	const j_real delta = fixeps*cij;
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		if(l[ivar] < delta)
		{
			l[ivar] = (l[ivar]*l[ivar] + delta*delta)/(2.0*delta);

			for(int k = 0; k < NVARS; k++)
			{
				dli[ivar][k] = ((2.0*(l[ivar]*dli[ivar][k]+delta*fixeps*dciji[k])*2.0*delta)
					- (l[ivar]*l[ivar]+delta*delta)*2.0*fixeps*dciji[k]) / (4.0*delta*delta);
				dlj[ivar][k] = ((2.0*(l[ivar]*dlj[ivar][k]+delta*fixeps*dcijj[k])*2.0*delta)
					- (l[ivar]*l[ivar]+delta*delta)*2.0*fixeps*dcijj[k]) / (4.0*delta*delta);
			}
		}
	}

	//> A_Roe * dU

	const j_real devn = vnj-vni, dep = pj-pi, derho = ur[0]-ul[0];

	j_real dderhoi[NVARS], dderhoj[NVARS];
	dderhoi[0] = -1.0; dderhoj[0] = 1.0;
	for(int k = 1; k < NVARS; k++)
	{
		dderhoi[k] = 0;
		dderhoj[k] = 0;
	}

	// product of eigenvalues and wave strengths

	j_real lalpha[NVARS]; j_real dlalphai[NVARS][NVARS], dlalphaj[NVARS][NVARS];
	const j_real cij4 = cij*cij*cij*cij;

	lalpha[0] = l[0]*(dep-rhoij*cij*devn)/(2.0*cij*cij);
	for(int k = 0; k < NVARS; k++)
	{
		dlalphai[0][k] = (( dli[0][k]*(dep-rhoij*cij*devn) +l[0]*(-dpi[k] - drhoiji[k]*cij*devn
			-rhoij*dciji[k]*devn-rhoij*cij*(-dvni[k])))*2.0*cij*cij - l[0]*(dep-rhoij*cij*devn) *
			4.0*cij*dciji[k] ) / (4.0*cij4);
		dlalphaj[0][k] = (( dlj[0][k]*(dep-rhoij*cij*devn) +l[0]*(dpj[k] - drhoijj[k]*cij*devn
			-rhoij*dcijj[k]*devn-rhoij*cij*dvnj[k]))*2.0*cij*cij - l[0]*(dep-rhoij*cij*devn) *
			4.0*cij*dcijj[k] ) / (4.0*cij4);
	}

	lalpha[1] = l[1]*(derho - dep/(cij*cij));
	for(int k = 0; k < NVARS; k++)
	{
		dlalphai[1][k] = dli[1][k]*(derho-dep/(cij*cij))+l[1]*(dderhoi[k] - ((-dpi[k])*cij*cij
			- dep*2.0*cij*dciji[k])/cij4);
		dlalphaj[1][k] = dlj[1][k]*(derho-dep/(cij*cij))+l[1]*(dderhoj[k] - (dpj[k]*cij*cij
			- dep*2.0*cij*dcijj[k])/cij4);
	}

	lalpha[2] = l[1]*rhoij;
	for(int k = 0; k < NVARS; k++)
	{
		dlalphai[2][k] = dli[1][k]*rhoij + l[1]*drhoiji[k];
		dlalphaj[2][k] = dlj[1][k]*rhoij + l[1]*drhoijj[k];
	}

	lalpha[3] = l[3]*(dep+rhoij*cij*devn)/(2.0*cij*cij);
	for(int k = 0; k < NVARS; k++)
	{
		dlalphai[3][k] = ((dli[3][k]*(dep+rhoij*cij*devn) + l[3]*(-dpi[k] +drhoiji[k]*cij*devn
			+rhoij*dciji[k]*devn+rhoij*cij*(-dvni[k])))*2.0*cij*cij - l[3]*(dep+rhoij*cij*devn)
			*4.0*cij*dciji[k]) / (4.0*cij4);
		dlalphaj[3][k] = ((dlj[3][k]*(dep+rhoij*cij*devn) + l[3]*(dpj[k] +drhoijj[k]*cij*devn
			+rhoij*dcijj[k]*devn +rhoij*cij*dvnj[k]))*2.0*cij*cij - l[3]*(dep+rhoij*cij*devn)
			*4.0*cij*dcijj[k]) / (4.0*cij4);
	}

	// dissipation terms

	j_real adu[NVARS]; j_real dadui[NVARS][NVARS], daduj[NVARS][NVARS];

	// un-c:
	adu[0] = lalpha[0];
	adu[1] = lalpha[0]*(vxij-cij*n[0]);
	adu[2] = lalpha[0]*(vyij-cij*n[1]);
	adu[3] = lalpha[0]*(Hij-cij*vnij);
	for(int k = 0; k < NVARS; k++)
	{
		dadui[0][k] = dlalphai[0][k];
		dadui[1][k] = dlalphai[0][k]*(vxij-cij*n[0]) + lalpha[0]*(dvxiji[k]-dciji[k]*n[0]);
		dadui[2][k] = dlalphai[0][k]*(vyij-cij*n[1]) + lalpha[0]*(dvyiji[k]-dciji[k]*n[1]);
		dadui[3][k] = dlalphai[0][k]*(Hij-cij*vnij)
						+ lalpha[0]*(dHiji[k]-dciji[k]*vnij-cij*dvniji[k]);

		daduj[0][k] = dlalphaj[0][k];
		daduj[1][k] = dlalphaj[0][k]*(vxij-cij*n[0]) + lalpha[0]*(dvxijj[k]-dcijj[k]*n[0]);
		daduj[2][k] = dlalphaj[0][k]*(vyij-cij*n[1]) + lalpha[0]*(dvyijj[k]-dcijj[k]*n[1]);
		daduj[3][k] = dlalphaj[0][k]*(Hij-cij*vnij)
						+ lalpha[0]*(dHijj[k]-dcijj[k]*vnij-cij*dvnijj[k]);
	}

	// un:
	adu[0] += lalpha[1];
	adu[1] += lalpha[1]*vxij +      lalpha[2]*(vxj-vxi - devn*n[0]);
	adu[2] += lalpha[1]*vyij +      lalpha[2]*(vyj-vyi - devn*n[1]);
	adu[3] += lalpha[1]*vm2ij/2.0 + lalpha[2] *(vxij*(vxj-vxi) +vyij*(vyj-vyi) -vnij*devn);
	for(int k = 0; k < NVARS; k++)
	{
		dadui[0][k] += dlalphai[1][k];
		dadui[1][k] += dlalphai[1][k]*vxij+lalpha[1]*dvxiji[k]
			+dlalphai[2][k]*(vxj-vxi-devn*n[0]) +lalpha[2]*(-dvxi[k]+dvni[k]*n[0]);
		dadui[2][k] += dlalphai[1][k]*vyij+lalpha[1]*dvyiji[k]
			+dlalphai[2][k]*(vyj-vyi-devn*n[1]) +lalpha[2]*(-dvyi[k]+dvni[k]*n[1]);
		dadui[3][k] += dlalphai[1][k]*vm2ij/2.0+lalpha[1]*dvm2iji[k]/2.0
			+dlalphai[2][k]*(vxij*(vxj-vxi)+vyij*(vyj-vyi)-vnij*devn)
			+ lalpha[2]*(dvxiji[k]*(vxj-vxi)+vxij*(-dvxi[k]) + dvyiji[k]*(vyj-vyi)+vyij*(-dvyi[k])
			-dvniji[k]*devn-vnij*(-dvni[k]));

		daduj[0][k] += dlalphaj[1][k];
		daduj[1][k] += dlalphaj[1][k]*vxij+lalpha[1]*dvxijj[k]
			+dlalphaj[2][k]*(vxj-vxi-devn*n[0]) +lalpha[2]*(dvxj[k]-dvnj[k]*n[0]);
		daduj[2][k] += dlalphaj[1][k]*vyij+lalpha[1]*dvyijj[k]
			+dlalphaj[2][k]*(vyj-vyi-devn*n[1]) +lalpha[2]*(dvyj[k]-dvnj[k]*n[1]);
		daduj[3][k] += dlalphaj[1][k]*vm2ij/2.0+lalpha[1]*dvm2ijj[k]/2.0
			+dlalphaj[2][k]*(vxij*(vxj-vxi)+vyij*(vyj-vyi)-vnij*devn)
			+ lalpha[2]*(dvxijj[k]*(vxj-vxi)+vxij*dvxj[k] + dvyijj[k]*(vyj-vyi)+vyij*dvyj[k]
			-dvnijj[k]*devn-vnij*dvnj[k]);
	}

	// un+c:
	adu[0] += lalpha[3];
	adu[1] += lalpha[3]*(vxij+cij*n[0]);
	adu[2] += lalpha[3]*(vyij+cij*n[1]);
	adu[3] += lalpha[3]*(Hij+cij*vnij);
	for(int k = 0; k < NVARS; k++)
	{
		dadui[0][k] += dlalphai[3][k];
		dadui[1][k] += dlalphai[3][k]*(vxij+cij*n[0]) + lalpha[3]*(dvxiji[k]+dciji[k]*n[0]);
		dadui[2][k] += dlalphai[3][k]*(vyij+cij*n[1]) + lalpha[3]*(dvyiji[k]+dciji[k]*n[1]);
		dadui[3][k] += dlalphai[3][k]*(Hij+cij*vnij)
						+ lalpha[3]*(dHiji[k]+dciji[k]*vnij+cij*dvniji[k]);

		daduj[0][k] += dlalphaj[3][k];
		daduj[1][k] += dlalphaj[3][k]*(vxij+cij*n[0]) + lalpha[3]*(dvxijj[k]+dcijj[k]*n[0]);
		daduj[2][k] += dlalphaj[3][k]*(vyij+cij*n[1]) + lalpha[3]*(dvyijj[k]+dcijj[k]*n[1]);
		daduj[3][k] += dlalphaj[3][k]*(Hij+cij*vnij)
						+ lalpha[3]*(dHijj[k]+dcijj[k]*vnij+cij*dvnijj[k]);
	}

	// get one-sided Jacobians
	/*j_real fi[4], fj[4];
	physics->getDirectionalFlux(ul,n,vni,pi,fi);
	physics->getDirectionalFlux(ur,n,vnj,pj,fj);*/
	physics->getJacobianDirectionalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianDirectionalFluxWrtConserved(ur, n, dfdr);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		//flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - adu[ivar]);
		for(int k = 0; k < NVARS; k++)
		{
			dfdl[ivar*NVARS+k] = - 0.5*(dfdl[ivar*NVARS+k] - dadui[ivar][k]);
			dfdr[ivar*NVARS+k] =   0.5*(dfdr[ivar*NVARS+k] - daduj[ivar][k]);
		}
	}
}

template <typename scalar, typename j_real>
HLLFlux<scalar,j_real>::HLLFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: RoeAverageBasedFlux<scalar,j_real>(analyticalflux)
{
}

template <typename scalar, typename j_real>
void HLLFlux<scalar,j_real>::get_flux(const scalar *const __restrict__ ul, const scalar *const __restrict__ ur,
		const scalar* const __restrict__ n, scalar *const __restrict__ flux) const
{
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0], pi);
	cj = physics->getSoundSpeed(ur[0], pj);

	//const scalar vxi = vi[0], vxj=vj[0], vyi=vi[1], vyj=vj[1];

	//> compute Roe-averages
	scalar Rij,rhoij,/*vxij,vyij,*/vm2ij,vnij,Hij,cij,vij[NDIM];
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);
	//vxij = vij[0]; vyij = vij[0];

	// Einfeldt estimate for signal speeds
	scalar sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	scalar sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const scalar sr0 = sr > 0 ? 0 : sr;
	const scalar sl0 = sl > 0 ? 0 : sl;

	// flux
	const scalar t1 = (sr0 - sl0)/(sr-sl); const scalar t2 = 1.0 - t1;
	const scalar t3 = 0.5*(sr*fabs(sl)-sl*fabs(sr))/(sr-sl);
	flux[0] = t1*vnj*ur[0] + t2*vni*ul[0]                     - t3*(ur[0]-ul[0]);
	flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
	flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
	flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi)           - t3*(ur[3]-ul[3]);
}

/** The linearization assumes `locally frozen' signal speeds.
 * According to Batten, Lechziner and Goldberg, this should be fine.
 */
template <typename scalar, typename j_real>
void HLLFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
                                          const j_real* const n,
                                          j_real *const __restrict dfdl,
                                          j_real *const __restrict dfdr) const
{
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0], pi);
	cj = physics->getSoundSpeed(ur[0], pj);

	//const j_real vxi = vi[0], vxj=vj[0], vyi=vi[1], vyj=vj[1];

	// compute Roe-averages
	j_real Rij,rhoij,/*vxij,vyij,*/vm2ij,vnij,Hij,cij,vij[NDIM];
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);

	// Einfeldt estimate for signal speeds
	j_real sr, sl;
	sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const j_real sr0 = sr > 0 ? 0 : sr;
	const j_real sl0 = sl > 0 ? 0 : sl;
	const j_real t1 = (sr0 - sl0)/(sr-sl);
	const j_real t2 = 1.0 - t1;
	const j_real t3 = 0.5*(sr*fabs(sl)-sl*fabs(sr))/(sr-sl);

	// get flux jacobians
	physics->getJacobianDirectionalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianDirectionalFluxWrtConserved(ur, n, dfdr);
	for(int i = 0; i < NVARS; i++)
		for(int j = 0; j < NVARS; j++)
		{
			// lower block
			dfdl[i*NVARS+j] = -t2*dfdl[i*NVARS+j];
			// upper block
			dfdr[i*NVARS+j] =  t1*dfdr[i*NVARS+j];
		}
	for(int i = 0; i < NVARS; i++) {
		// lower:
		dfdl[i*NVARS+i] = dfdl[i*NVARS+i] - t3;
		// upper:
		dfdr[i*NVARS+i] = dfdr[i*NVARS+i] - t3;
	}
}

template <typename scalar, typename j_real>
HLLCFlux<scalar,j_real>::HLLCFlux(const IdealGasPhysics<scalar> *const analyticalflux)
	: RoeAverageBasedFlux<scalar>(analyticalflux)
{
}

template <typename scalar, typename j_real>
inline
void HLLCFlux<scalar,j_real>::getStarState(const scalar u[NVARS], const scalar n[NDIM],
	const scalar vn, const scalar p,
	const scalar ss, const scalar sm,
	scalar *const __restrict ustr) const
{
	const scalar pstar = u[0]*(vn-ss)*(vn-sm) + p;
	ustr[0] = u[0] * (ss - vn)/(ss-sm);
	ustr[1] = ( (ss-vn)*u[1] + (pstar-p)*n[0] )/(ss-sm);
	ustr[2] = ( (ss-vn)*u[2] + (pstar-p)*n[1] )/(ss-sm);
	ustr[3] = ( (ss-vn)*u[3] - p*vn + pstar*sm )/(ss-sm);
}

template <typename scalar, typename j_real>
inline
void HLLCFlux<scalar,j_real>::getStarStateAndJacobian(const j_real u[NVARS], const j_real n[NDIM],
                                                      const j_real vn, const j_real p,
                                                      const j_real ss, const j_real sm,
                                                      const j_real dvn[NVARS], const j_real dp[NVARS],
                                                      const j_real dssi[NDIM], const j_real dsmi[NDIM],
                                                      const j_real dssj[NDIM], const j_real dsmj[NDIM],
                                                      j_real ustr[NVARS],
                                                      j_real dustri[NVARS][NVARS] ,
                                                      j_real dustrj[NVARS][NVARS] ) const
{
	const j_real pstar = u[0]*(vn-ss)*(vn-sm) + p;

	j_real dpsi[NVARS], dpsj[NVARS];

	dpsi[0] = (vn-ss)*(vn-sm) +u[0]*(dvn[0]-dssi[0])*(vn-sm)
		+u[0]*(vn-ss)*(dvn[0]-dsmi[0]) + dp[0];
	dpsj[0] = u[0]*((-dssj[0])*(vn-sm) + (vn-ss)*(-dsmj[0]));
	for(int k = 1; k < NVARS; k++)
	{
		dpsi[k] = u[0]*((dvn[k]-dssi[k])*(vn-sm)+(vn-ss)*(dvn[k]-dsmi[k])) + dp[k];
		dpsj[k] = u[0]*((-dssj[k])*(vn-sm)+(vn-ss)*(-dsmj[k]));
	}

	ustr[0] = u[0] * (ss - vn)/(ss-sm);

	dustri[0][0]=u[0]*((dssi[0]-dvn[0])*(ss-sm)-(ss-vn)*(dssi[0]-dsmi[0]))/((ss-sm)*(ss-sm))
		+ (ss-vn)/(ss-sm);
	dustrj[0][0]=u[0]*(dssj[0]*(ss-sm)-(ss-vn)*(dssj[0]-dsmj[0])) / ((ss-sm)*(ss-sm));
	for(int k = 1; k < NVARS; k++)
	{
		dustri[0][k]=u[0]*((dssi[k]-dvn[k])*(ss-sm)-(ss-vn)*(dssi[k]-dsmi[k]))
			/ ((ss-sm)*(ss-sm));
		dustrj[0][k]=u[0]*(dssj[k]*(ss-sm)-(ss-vn)*(dssj[k]-dsmj[k])) / ((ss-sm)*(ss-sm));
	}

	ustr[1] = ( (ss-vn)*u[1] + (pstar-p)*n[0] )/(ss-sm);

	for(int k = 0; k < NVARS; k++)
	{
		if(k == 1) continue;
		dustri[1][k]= ( ((dssi[k]-dvn[k])*u[1] + (dpsi[k]-dp[k])*n[0])*(ss-sm)
			- ((ss-vn)*u[1]+(pstar-p)*n[0])*(dssi[k]-dsmi[k]) )/((ss-sm)*(ss-sm));
		dustrj[1][k]= ( (dssj[k]*u[1] + dpsj[k]*n[0])*(ss-sm)
			- ((ss-vn)*u[1]+(pstar-p)*n[0])*(dssj[k]-dsmj[k]) )/((ss-sm)*(ss-sm));
	}
	dustri[1][1]= ( ((dssi[1]-dvn[1])*u[1]+(ss-vn) + (dpsi[1]-dp[1])*n[0])*(ss-sm)
			- ((ss-vn)*u[1]+(pstar-p)*n[0])*(dssi[1]-dsmi[1]) )/((ss-sm)*(ss-sm));

	dustrj[1][1]= ( (dssj[1]*u[1] + dpsj[1]*n[0])*(ss-sm)
		- ((ss-vn)*u[1]+(pstar-p)*n[0])*(dssj[1]-dsmj[1]) )/((ss-sm)*(ss-sm));

	ustr[2] = ( (ss-vn)*u[2] + (pstar-p)*n[1] )/(ss-sm);

	for(int k = 0; k < NVARS; k++)
	{
		if(k == 2) continue;
		dustri[2][k]= ( ((dssi[k]-dvn[k])*u[2] + (dpsi[k]-dp[k])*n[1])*(ss-sm)
			- ((ss-vn)*u[2]+(pstar-p)*n[1])*(dssi[k]-dsmi[k]) )/((ss-sm)*(ss-sm));
		dustrj[2][k]= ( (dssj[k]*u[2] + dpsj[k]*n[1])*(ss-sm)
			- ((ss-vn)*u[2]+(pstar-p)*n[1])*(dssj[k]-dsmj[k]) )/((ss-sm)*(ss-sm));
	}

	dustri[2][2]= ( ((dssi[2]-dvn[2])*u[2]+(ss-vn) + (dpsi[2]-dp[2])*n[1])*(ss-sm)
			- ((ss-vn)*u[2]+(pstar-p)*n[1])*(dssi[2]-dsmi[2]) )/((ss-sm)*(ss-sm));

	dustrj[2][2]= ( (dssj[2]*u[2] + dpsj[2]*n[1])*(ss-sm)
		- ((ss-vn)*u[2]+(pstar-p)*n[1])*(dssj[2]-dsmj[2]) )/((ss-sm)*(ss-sm));

	ustr[3] = ( (ss-vn)*u[3] - p*vn + pstar*sm )/(ss-sm);

	for(int k = 0; k < NVARS-1; k++)
	{
		dustri[3][k]= ( ((dssi[k]-dvn[k])*u[3] -dp[k]*vn-p*dvn[k]
			+dpsi[k]*sm+pstar*dsmi[k]) * (ss-sm)
			- ((ss-vn)*u[3]-p*vn+pstar*sm)*(dssi[k]-dsmi[k]) )/((ss-sm)*(ss-sm));

		dustrj[3][k]= ( (dssj[k]*u[3] + dpsj[k]*sm+pstar*dsmj[k])*(ss-sm)
			- ((ss-vn)*u[3]-p*vn+pstar*sm)*(dssj[k]-dsmj[k]) )/((ss-sm)*(ss-sm));
	}

	dustri[3][3]= ( ((dssi[3]-dvn[3])*u[3]+(ss-vn) -dp[3]*vn-p*dvn[3]
		+dpsi[3]*sm+pstar*dsmi[3]) * (ss-sm)
		- ((ss-vn)*u[3]-p*vn+pstar*sm)*(dssi[3]-dsmi[3]) )/((ss-sm)*(ss-sm));

	dustrj[3][3]= ( (dssj[3]*u[3] + dpsj[3]*sm+pstar*dsmj[3])*(ss-sm)
		- ((ss-vn)*u[3]-p*vn+pstar*sm)*(dssj[3]-dsmj[3]) )/((ss-sm)*(ss-sm));
}

/** \todo See if the implementation can be tweaked to reduce round-off errors.
 */
template <typename scalar, typename j_real>
void HLLCFlux<scalar,j_real>::get_flux(const scalar *const ul, const scalar *const ur,
                                const scalar* const n,
                                scalar *const __restrict flux) const
{
	scalar vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0], pi);
	cj = physics->getSoundSpeed(ur[0], pj);

	// compute Roe-averages
	scalar Rij,rhoij,vij[NDIM],vm2ij,vnij,Hij,cij;
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);

	// estimate signal speeds
	scalar sr, sl;
	sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const scalar sm = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj )
		/ ( ur[0]*(sr-vnj) - ul[0]*(sl-vni) );

	// compute fluxes

	if(sl > 0)
		physics->getDirectionalFlux(ul,n,vni,pi,flux);

	else if(sl <= 0 && sm > 0)
	{
		physics->getDirectionalFlux(ul,n,vni,pi,flux);

		scalar ulstr[NVARS];
		getStarState(ul,n,vni,pi,sl,sm,ulstr);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sl * ( ulstr[ivar] - ul[ivar]);
	}
	else if(sm <= 0 && sr >= 0)
	{
		physics->getDirectionalFlux(ur,n,vnj,pj,flux);

		scalar urstr[NVARS];
		getStarState(ur,n,vnj,pj,sr,sm,urstr);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sr * ( urstr[ivar] - ur[ivar]);
	}
	else
		physics->getDirectionalFlux(ur,n,vnj,pj,flux);
}

template <typename scalar, typename j_real>
void HLLCFlux<scalar,j_real>::get_jacobian(const j_real *const ul, const j_real *const ur,
                                           const j_real* const n,
                                           j_real *const __restrict dfdl,
                                           j_real *const __restrict dfdr) const
{
	j_real vi[NDIM], vj[NDIM], vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vj, vnj, pj, Hj);
	ci = physics->getSoundSpeed(ul[0], pi);
	cj = physics->getSoundSpeed(ur[0], pj);

	const j_real vxi = vi[0], vxj=vj[0], vyi=vi[1], vyj=vj[1];

	// compute Roe-averages
	j_real Rij,rhoij,vm2ij,vnij,Hij,cij,vij[NDIM];
	getRoeAverages(ul,ur,n,vi,Hi,vj,Hj, Rij,rhoij,vij,vm2ij,vnij,Hij,cij);

	//> Derivatives of the above variables

	j_real dpi[NVARS], dpj[NVARS], dvni[NVARS], dvnj[NVARS], dvi[NDIM*NVARS], dvj[NDIM*NVARS],
	dHi[NVARS], dHj[NVARS], dci[NVARS], dcj[NVARS],
	dRiji[NVARS], dRijj[NVARS], dvxiji[NVARS], dvyiji[NVARS], dvxijj[NVARS], dvyijj[NVARS],
	dvniji[NVARS], dvnijj[NVARS], dvm2iji[NVARS], dvm2ijj[NVARS], dciji[NVARS], dcijj[NVARS],
	drhoiji[NVARS], drhoijj[NVARS], dHiji[NVARS], dHijj[NVARS];
	for(int k = 0; k < NVARS; k++)
	{
		dpi[k] = dpj[k] = dHi[k] = dHj[k] = dci[k] = dcj[k] = 0;
		dvni[k] = dvnj[k] = 0;
		for(int j = 0; j < NDIM; j++) {
			dvi[j*NVARS+k] = 0;
			dvj[j*NVARS+k] = 0;
		}
	}

	physics->getJacobianVarsWrtConserved(ul,n,dvi,dvni,dpi,dHi);
	physics->getJacobianVarsWrtConserved(ur,n,dvj,dvnj,dpj,dHj);
	physics->getJacobianSoundSpeed(ul[0],pi,dpi,ci,dci);
	physics->getJacobianSoundSpeed(ur[0],pj,dpj,cj,dcj);

	j_real dvxi[NVARS], dvxj[NVARS], dvyi[NVARS],dvyj[NVARS];
	for(int k = 0; k < NVARS; k++) {
		dvxi[k] = dvi[k]; dvxj[k] = dvj[k];
		dvyi[k] = dvi[NVARS+k]; dvyj[k] = dvj[NVARS+k];
	}

	getJacobiansRoeAveragesWrtConserved(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj,dvxi,dvyi,dHi,dvxj,dvyj,dHj,
		dRiji, drhoiji, dvxiji, dvyiji, dvm2iji, dvniji, dHiji, dciji,
		dRijj, drhoijj, dvxijj, dvyijj, dvm2ijj, dvnijj, dHijj, dcijj);

	// estimate signal speeds
	j_real sr, sl, dsli[NVARS], dslj[NVARS], dsri[NVARS], dsrj[NVARS];
	sl = vni - ci;
	for(int k = 0; k < NVARS; k++) {
		dsli[k] = dvni[k] - dci[k];
		dslj[k] = 0;
	}
	if (sl > vnij-cij) {
		sl = vnij-cij;
		for(int k = 0; k < NVARS; k++) {
			dsli[k] = dvniji[k] - dciji[k];
			dslj[k] = dvnijj[k] - dcijj[k];
		}
	}

	sr = vnj+cj;
	for(int k = 0; k < NVARS; k++) {
		dsri[k] = 0;
		dsrj[k] = dvnj[k] + dcj[k];
	}
	if(sr < vnij+cij) {
		sr = vnij+cij;
		for(int k = 0; k < NVARS; k++) {
			dsri[k] = dvniji[k] + dciji[k];
			dsrj[k] = dvnijj[k] + dcijj[k];
		}
	}

	const j_real num = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj );
	const j_real denom = (ur[0]*(sr-vnj) - ul[0]*(sl-vni));

	const j_real sm = num / denom;
	j_real dsmi[NVARS], dsmj[NVARS];

	dsmi[0]= ( (ur[0]*vnj*dsri[0] -vni*(sl-vni)-ul[0]*dvni[0]*(sl-vni)-ul[0]*vni*(dsli[0]-dvni[0])
		+ dpi[0] )*denom
		-num*(ur[0]*dsri[0] - (sl-vni)-ul[0]*(dsli[0]-dvni[0])) ) / (denom*denom);

	dsmj[0]= ( (vnj*(sr-vnj)+ur[0]*dvnj[0]*(sr-vnj)+ur[0]*vnj*(dsrj[0]-dvnj[0]) -ul[0]*vni*dslj[0]
		- dpj[0])*denom
		-num*((sr-vnj)+ur[0]*(dsrj[0]-dvnj[0]) - ul[0]*dslj[0]) ) / (denom*denom);

	for(int k = 1; k < NVARS; k++) {
		dsmi[k]= ( (ur[0]*vnj*dsri[k] - ul[0]*(dvni[k]*(sl-vni)+vni*(dsli[k]-dvni[k])) +dpi[k])
		  * denom - num *(ur[0]*dsri[k] -ul[0]*(dsli[k]-dvni[k])) ) / (denom*denom);

		dsmj[k]= ( (ur[0]*(dvnj[k]*(sr-vnj)+vnj*(dsrj[k]-dvnj[k])) -ul[0]*vni*dslj[k] -dpj[k])
			* denom - num * (ur[0]*(dsrj[k]-dvnj[k]) - ul[0]*dslj[k]) ) / (denom*denom);
	}

	// compute fluxes

	if(sl > 0)
	{
		//physics->getDirectionalFlux(ul,n,vni,pi,flux);

		physics->getJacobianDirectionalFluxWrtConserved(ul,n,dfdl);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdr[k] = 0;
	}
	else if(sl <= 0 && sm > 0)
	{
		//physics->getDirectionalFlux(ul,n,vni,pi,flux);

		physics->getJacobianDirectionalFluxWrtConserved(ul,n,dfdl);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdr[k] = 0;

		j_real ulstr[NVARS], dulstri[NVARS][NVARS], dulstrj[NVARS][NVARS];
		getStarStateAndJacobian(ul,n,vni,pi,sl,sm,dvni,dpi,dsli,dsmi,dslj,dsmj,
				ulstr,dulstri,dulstrj);

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			//flux[ivar] += sl * ( ulstr[ivar] - ul[ivar]);
			for(int k = 0; k < NVARS; k++)
			{
				dfdl[ivar*NVARS+k] += dsli[k]*(ulstr[ivar]-ul[ivar])
					+ sl*(dulstri[ivar][k] - (ivar==k ? 1.0 : 0.0));
				dfdr[ivar*NVARS+k] += dslj[k]*(ulstr[ivar]-ul[ivar]) + sl*dulstrj[ivar][k];
			}
		}
	}
	else if(sm <= 0 && sr >= 0)
	{
		//physics->getDirectionalFlux(ur,n,vnj,pj,flux);

		physics->getJacobianDirectionalFluxWrtConserved(ur,n,dfdr);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdl[k] = 0;

		j_real urstr[NVARS], durstri[NVARS][NVARS], durstrj[NVARS][NVARS];
		getStarStateAndJacobian(ur,n,vnj,pj,sr,sm,dvnj,dpj,dsrj,dsmj,dsri,dsmi,
				urstr,durstrj,durstri);

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			//flux[ivar] += sr * ( urstr[ivar] - ur[ivar]);

			for(int k = 0; k < NVARS; k++) {
				dfdl[ivar*NVARS+k] += dsri[k]*(urstr[ivar]-ur[ivar]) +sr*durstri[ivar][k];
				dfdr[ivar*NVARS+k] += dsrj[k]*(urstr[ivar]-ur[ivar])
										+ sr*(durstrj[ivar][k] - (ivar==k ? 1.0:0.0));
			}
		}
	}
	else
	{
		//physics->getDirectionalFlux(ur,n,vnj,pj,flux);

		physics->getJacobianDirectionalFluxWrtConserved(ur,n,dfdr);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdl[k] = 0;
	}

	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

template class LocalLaxFriedrichsFlux<freal>;
template class VanLeerFlux<freal>;
template class AUSMFlux<freal>;
template class AUSMPlusFlux<freal>;
template class RoeFlux<freal>;
template class HLLFlux<freal>;
template class HLLCFlux<freal>;

//CHANGE HERE
#ifdef USE_ADOLC
//template class LocalLaxFriedrichsFlux<adouble>;
template class VanLeerFlux<adouble>;
//template class AUSMFlux<adouble>;
template class AUSMPlusFlux<adouble>;
//template class RoeFlux<adouble>;
//template class HLLFlux<adouble>;
//template class HLLCFlux<adouble>;
#endif

} // end namespace fvens

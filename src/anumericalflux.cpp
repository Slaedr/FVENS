/** \file anumericalflux.cpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015, June 2017, September-October 2017
 */

/* Tapenade notes:
 * No consts
 * #defines work but get replaced
 */

#include "anumericalflux.hpp"

namespace acfd {

InviscidFlux::InviscidFlux(const IdealGasPhysics *const phyctx) 
	: physics(phyctx), g{phyctx->g}
{ }

void InviscidFlux::get_jacobian(const a_real *const uleft, const a_real *const uright, 
		const a_real* const n, 
		a_real *const dfdl, a_real *const dfdr)
{ }

InviscidFlux::~InviscidFlux()
{ }

LocalLaxFriedrichsFlux::LocalLaxFriedrichsFlux(const IdealGasPhysics *const analyticalflux)
	: InviscidFlux(analyticalflux)
{ }

void LocalLaxFriedrichsFlux::get_flux(const a_real *const ul, 
		const a_real *const ur, 
		const a_real *const n, 
		a_real *const __restrict flux)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	const a_real eig = 
		std::fabs(vni)+ci > std::fabs(vnj)+cj ? std::fabs(vni)+ci : std::fabs(vnj)+cj;
	
	flux[0] = 0.5*( ul[0]*vni + ur[0]*vnj - eig*(ur[0]-ul[0]) );
	flux[1] = 0.5*( vni*ul[1]+pi*n[0] + vnj*ur[1]+pj*n[0] - eig*(ur[1]-ul[1]) );
	flux[2] = 0.5*( vni*ul[2]+pi*n[1] + vnj*ur[2]+pj*n[1] - eig*(ur[2]-ul[2]) );
	flux[3] = 0.5*( vni*(ul[3]+pi) + vnj*(ur[3]+pj) - eig*(ur[3] - ul[3]) );
}

/** Jacobian with frozen spectral radius
 */
void LocalLaxFriedrichsFlux::get_jacobian(const a_real *const ul, const a_real *const ur,
		const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj, eig;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	// max eigenvalue
	if( std::fabs(vni)+ci >= std::fabs(vnj)+cj )
	{
		eig = std::fabs(vni)+ci;
	}
	else
	{
		eig = std::fabs(vnj)+cj;
	}

	// get flux jacobians
	physics->getJacobianNormalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianNormalFluxWrtConserved(ur, n, dfdr);

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

// full linearization, inspite of the name; no better than frozen version
void LocalLaxFriedrichsFlux::get_frozen_jacobian(const a_real *const ul, 
		const a_real *const ur,
		const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj, eig;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	// max eigenvalue
	bool leftismax;
	if( std::fabs(vni)+ci >= std::fabs(vnj)+cj )
	{
		eig = std::fabs(vni)+ci;
		leftismax = true;
	}
	else
	{
		eig = std::fabs(vnj)+cj;
		leftismax = false;
	}

	// get flux jacobians
	physics->getJacobianNormalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianNormalFluxWrtConserved(ur, n, dfdr);

	// linearization of the dissipation term
	
	const a_real ctermi = 0.5 / 
		std::sqrt( g*(g-1)/ul[0]* (ul[3]-(ul[1]*ul[1]+ul[2]*ul[2])/(2*ul[0])) );
	const a_real ctermj = 0.5 / 
		std::sqrt( g*(g-1)/ur[0]* (ur[3]-(ur[1]*ur[1]+ur[2]*ur[2])/(2*ur[0])) );
	a_real dedu[NVARS];
	
	if(leftismax) {
		dedu[0] = -std::fabs(vni/(ul[0]*ul[0])) + ctermi*g*(g-1)*( -ul[3]/(ul[0]*ul[0]) 
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
		dedu[0] = -std::fabs(vnj/(ur[0]*ur[0])) + ctermj*g*(g-1)*( -ur[3]/(ur[0]*ur[0]) 
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

VanLeerFlux::VanLeerFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{
}

void VanLeerFlux::get_flux(const a_real *const ul, const a_real *const ur,
		const a_real* const n, a_real *const __restrict flux)
{
	a_real fiplus[NVARS], fjminus[NVARS];

	const a_real nx = n[0];
	const a_real ny = n[1];

	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	//Normal mach numbers
	const a_real Mni = vni/ci;
	const a_real Mnj = vnj/cj;

	//Calculate split fluxes
	if(Mni < -1.0)
		for(int i = 0; i < NVARS; i++)
			fiplus[i] = 0;
	else if(Mni > 1.0)
		physics->getNormalFluxEfficiently(ul,n,vni,pi,fiplus);
	else
	{
		const a_real vmags = pow(ul[1]/ul[0], 2) + pow(ul[2]/ul[0], 2);
		fiplus[0] = ul[0]*ci*pow(Mni+1, 2)/4.0;
		fiplus[1] = fiplus[0] * (ul[1]/ul[0] + nx*(2.0*ci - vni)/g);
		fiplus[2] = fiplus[0] * (ul[2]/ul[0] + ny*(2.0*ci - vni)/g);
		fiplus[3] = fiplus[0] * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
	}

	if(Mnj > 1.0)
		for(int i = 0; i < NVARS; i++)
			fjminus[i] = 0;
	else if(Mnj < -1.0)
		physics->getNormalFluxEfficiently(ur,n,vnj,pj,fjminus);
	else
	{
		const a_real vmags = pow(ur[1]/ur[0], 2) + pow(ur[2]/ur[0], 2);
		fjminus[0] = -ur[0]*cj*pow(Mnj-1, 2)/4.0;
		fjminus[1] = fjminus[0] * (ur[1]/ur[0] + nx*(-2.0*cj - vnj)/g);
		fjminus[2] = fjminus[0] * (ur[2]/ur[0] + ny*(-2.0*cj - vnj)/g);
		fjminus[3] = fjminus[0] * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
	}

	//Update the flux vector
	for(int i = 0; i < NVARS; i++)
		flux[i] = fiplus[i] + fjminus[i];
}

void VanLeerFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	std::cout << " ! VanLeerFlux: Not implemented!\n";
}

AUSMFlux::AUSMFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{ }

void AUSMFlux::get_flux(const a_real *const ul, const a_real *const ur,
		const a_real* const n, a_real *const __restrict flux)
{
	a_real ML, MR, pL, pR;
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	const a_real Mni = vni/ci, Mnj = vnj/cj;
	
	// split non-dimensional convection speeds (ie split Mach numbers) and split pressures
	if(std::fabs(Mni) <= 1.0)
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
	
	if(std::fabs(Mnj) <= 1.0) {
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
	const a_real Mhalf = ML+MR;
	const a_real phalf = pL+pR;

	// Fluxes
	flux[0] = Mhalf/2.0*(ul[0]*ci+ur[0]*cj) -std::fabs(Mhalf)/2.0*(ur[0]*cj-ul[0]*ci);
	flux[1] = Mhalf/2.0*(ul[1]*ci+ur[1]*cj) -std::fabs(Mhalf)/2.0*(ur[1]*cj-ul[1]*ci) + phalf*n[0];
	flux[2] = Mhalf/2.0*(ul[2]*ci+ur[2]*cj) -std::fabs(Mhalf)/2.0*(ur[2]*cj-ul[2]*ci) + phalf*n[1];
	flux[3] = Mhalf/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) 
		-std::fabs(Mhalf)/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi));
}

void AUSMFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	using std::fabs;
	a_real ML, MR;
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	const a_real Mni = vni/ci, Mnj = vnj/cj;

	a_real dpi[NVARS], dci[NVARS], dpj[NVARS], dcj[NVARS], dmni[NVARS], dmnj[NVARS];
	a_real dML[NVARS], dMR[NVARS], dpL[NVARS], dpR[NVARS];
	for(int i = 0; i < NVARS; i++) {
		dpi[i] = 0; dci[i] = 0; dpj[i] = 0; dcj[i] = 0; dmni[i] = 0; dmnj[i] = 0;
		dML[i] = dMR[i] = dpL[i] = dpR[i] = 0;
	}
	physics->getJacobianPressureWrtConserved(ul, dpi);
	physics->getJacobianPressureWrtConserved(ur, dpj);
	physics->getJacobianSoundSpeedWrtConserved(ul, dci);
	physics->getJacobianSoundSpeedWrtConserved(ur, dcj);

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
	const a_real Mhalf = ML+MR;
	//const a_real phalf = pL+pR;

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

	// x-momentum
	//flux[1] = Mhalf/2.0*(ul[1]*ci+ur[1]*cj) -fabs(Mhalf)/2.0*(ur[1]*cj-ul[1]*ci) + phalf*n[0];

	dfdl[NVARS+1] = dML[1]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*(ci+ul[1]*dci[1]) -
		( (Mhalf>=0? 1.0:-1.0)*dML[1]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*(-ci-ul[1]*dci[1]) )
		+ dpL[1]*n[0];
	dfdr[NVARS+1] = dMR[1]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*(cj+ur[1]*dcj[1]) -
		( (Mhalf>=0? 1.0:-1.0)*dMR[1]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*(cj+ur[1]*dcj[1]) )
		+ dpR[1]*n[0];
	for(int k = 0; k < NVARS; k++)
	{
		if(k == 1) continue;
		dfdl[NVARS+k] = dML[k]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*ul[1]*dci[k] - 
			( (Mhalf>=0?1.0:-1.0)*dML[k]/2.0*(ur[1]*cj-ul[1]*ci) - fabs(Mhalf)/2.0*ul[1]*dci[k] )
			+ dpL[k]*n[0];
		dfdr[NVARS+k] = dMR[k]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*ur[1]*dcj[k] - 
			( (Mhalf>=0?1.0:-1.0)*dMR[k]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*ur[1]*dcj[k] )
			+ dpR[k]*n[0];
	}

	// y-momentum
	//flux[2] = Mhalf/2.0*(ul[2]*ci+ur[2]*cj) -fabs(Mhalf)/2.0*(ur[2]*cj-ul[2]*ci) + phalf*n[1];

	dfdl[2*NVARS+2] = dML[2]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*(ci+ul[2]*dci[2]) -
		( (Mhalf>=0? 1.0:-1.0)*dML[2]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*(-ci-ul[2]*dci[2]) )
		+ dpL[2]*n[1];
	dfdr[2*NVARS+2] = dMR[2]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*(cj+ur[2]*dcj[2]) -
		( (Mhalf>=0? 1.0:-1.0)*dMR[2]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*(cj+ur[2]*dcj[2]) )
		+ dpR[2]*n[1];
	for(int k = 0; k < NVARS; k++)
	{
		if(k == 2) continue;
		dfdl[2*NVARS+k] = dML[k]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*ul[2]*dci[k] - 
			( (Mhalf>=0?1.0:-1.0)*dML[k]/2.0*(ur[2]*cj-ul[2]*ci) - fabs(Mhalf)/2.0*ul[2]*dci[k] )
			+ dpL[k]*n[1];
		dfdr[2*NVARS+k] = dMR[k]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*ur[2]*dcj[k] - 
			( (Mhalf>=0?1.0:-1.0)*dMR[k]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*ur[2]*dcj[k] )
			+ dpR[k]*n[1];
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

void AUSMFlux::get_flux_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, 
		a_real *const __restrict flux, a_real *const dfdl, a_real *const dfdr)
{
	using std::fabs;
	a_real ML, MR, pL, pR;
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);

	const a_real Mni = vni/ci, Mnj = vnj/cj;

	a_real dpi[NVARS], dci[NVARS], dpj[NVARS], dcj[NVARS], dmni[NVARS], dmnj[NVARS];
	a_real dML[NVARS], dMR[NVARS], dpL[NVARS], dpR[NVARS];
	for(int i = 0; i < NVARS; i++) {
		dpi[i] = 0; dci[i] = 0; dpj[i] = 0; dcj[i] = 0; dmni[i] = 0; dmnj[i] = 0;
		dML[i] = dMR[i] = dpL[i] = dpR[i] = 0;
	}
	physics->getJacobianPressureWrtConserved(ul, dpi);
	physics->getJacobianPressureWrtConserved(ur, dpj);
	physics->getJacobianSoundSpeedWrtConserved(ul, dci);
	physics->getJacobianSoundSpeedWrtConserved(ur, dcj);

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

		pL = ML*pi*(2.0-Mni);
		for(int k = 0; k < NVARS; k++)
			dpL[k] = dML[k]*pi*(2.0-Mni) + ML*dpi[k]*(2.0-Mni) - ML*pi*dmni[k];
	}
	else if(Mni < -1.0) {
		ML = 0;
		pL = 0;
	}
	else {
		ML = Mni;
		pL = pi;
		for(int k = 0; k < NVARS; k++) {
			dML[k] = dmni[k];
			dpL[k] = dpi[k];
		}
	}
	
	if(fabs(Mnj) <= 1.0) {
		MR = -0.25*(Mnj-1)*(Mnj-1);
		for(int k = 0; k < NVARS; k++)
			dMR[k] = -0.5*(Mnj-1)*dmnj[k];

		pR = -MR*pj*(2.0+Mnj);
		for(int k = 0; k < NVARS; k++)
			dpR[k] = -dMR[k]*pj*(2.0+Mnj) - MR*dpj[k]*(2.0+Mnj) - MR*pj*dmnj[k];
	}
	else if(Mnj < -1.0) {
		MR = Mnj;
		pR = pj;
		for(int k = 0; k < NVARS; k++) {
			dMR[k] = dmnj[k];
			dpR[k] = dpj[k];
		}
	}
	else {
		MR = 0;
		pR = 0;
	}
	
	// Interface convection speed and pressure
	const a_real Mhalf = ML+MR;
	const a_real phalf = pL+pR;

	/* Note that derivative of Mhalf w.r.t. ul is dML and that w.r.t. ur dMR,
	 * and similarly for phalf.
	 */

	// mass flux
	flux[0] = Mhalf/2.0*(ul[0]*ci+ur[0]*cj) -fabs(Mhalf)/2.0*(ur[0]*cj-ul[0]*ci);
	
	dfdl[0] = dML[0]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*(ci+ul[0]*dci[0])
		-( (Mhalf>0 ? 1.0 : -1.0)*dML[0]/2.0*(ur[0]*cj-ul[0]*ci) 
				+ fabs(Mhalf)/2.0*(-ci-ul[0]*dci[0]) );
	
	dfdr[0] = dMR[0]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*(cj+ur[0]*dcj[0])
		-( (Mhalf>0 ? 1.0 : -1.0)*dMR[0]/2.0*(ur[0]*cj-ul[0]*ci) 
				+ fabs(Mhalf)/2.0*(cj+ur[0]*dcj[0]) );
	
	for(int k = 1; k < NVARS; k++) 
	{
	  dfdl[k] = dML[k]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*ul[0]*dci[k]
		  - ( (Mhalf>0 ? 1.0:-1.0)*dML[k]/2.0*(ur[0]*cj-ul[0]*ci) - fabs(Mhalf)/2.0*ul[0]*dci[k] );
	  dfdr[k] = dMR[k]/2.0*(ul[0]*ci+ur[0]*cj) + Mhalf/2.0*ur[0]*dcj[k]
		  - ( (Mhalf>0 ? 1.0:-1.0)*dMR[k]/2.0*(ur[0]*cj-ul[0]*ci) + fabs(Mhalf)/2.0*ur[0]*dcj[k] );
	}

	// x-momentum
	flux[1] = Mhalf/2.0*(ul[1]*ci+ur[1]*cj) -fabs(Mhalf)/2.0*(ur[1]*cj-ul[1]*ci) + phalf*n[0];

	dfdl[NVARS+1] = dML[1]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*(ci+ul[1]*dci[1]) -
		( (Mhalf>0? 1.0:-1.0)*dML[1]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*(-ci-ul[1]*dci[1]) )
		+ dpL[1]*n[0];
	dfdr[NVARS+1] = dMR[1]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*(cj+ur[1]*dcj[1]) -
		( (Mhalf>0? 1.0:-1.0)*dMR[1]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*(cj+ur[1]*dcj[1]) )
		+ dpR[1]*n[0];
	for(int k = 0; k < NVARS; k++)
	{
		if(k == 1) continue;
		dfdl[NVARS+k] = dML[k]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*ul[1]*dci[k] - 
			( (Mhalf>0?1.0:-1.0)*dML[k]/2.0*(ur[1]*cj-ul[1]*ci) - fabs(Mhalf)/2.0*ul[1]*dci[k] )
			+ dpL[k]*n[0];
		dfdr[NVARS+k] = dMR[k]/2.0*(ul[1]*ci+ur[1]*cj) + Mhalf/2.0*ur[1]*dcj[k] - 
			( (Mhalf>0?1.0:-1.0)*dMR[k]/2.0*(ur[1]*cj-ul[1]*ci) + fabs(Mhalf)/2.0*ur[1]*dcj[k] )
			+ dpR[k]*n[0];
	}

	// y-momentum
	flux[2] = Mhalf/2.0*(ul[2]*ci+ur[2]*cj) -fabs(Mhalf)/2.0*(ur[2]*cj-ul[2]*ci) + phalf*n[1];

	dfdl[2*NVARS+2] = dML[2]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*(ci+ul[2]*dci[2]) -
		( (Mhalf>0? 1.0:-1.0)*dML[2]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*(-ci-ul[2]*dci[2]) )
		+ dpL[2]*n[1];
	dfdr[2*NVARS+2] = dMR[2]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*(cj+ur[2]*dcj[2]) -
		( (Mhalf>0? 1.0:-1.0)*dMR[2]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*(cj+ur[2]*dcj[2]) )
		+ dpR[2]*n[1];
	for(int k = 0; k < NVARS; k++)
	{
		if(k == 2) continue;
		dfdl[2*NVARS+k] = dML[k]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*ul[2]*dci[k] - 
			( (Mhalf>0?1.0:-1.0)*dML[k]/2.0*(ur[2]*cj-ul[2]*ci) - fabs(Mhalf)/2.0*ul[2]*dci[k] )
			+ dpL[k]*n[1];
		dfdr[2*NVARS+k] = dMR[k]/2.0*(ul[2]*ci+ur[2]*cj) + Mhalf/2.0*ur[2]*dcj[k] - 
			( (Mhalf>0?1.0:-1.0)*dMR[k]/2.0*(ur[2]*cj-ul[2]*ci) + fabs(Mhalf)/2.0*ur[2]*dcj[k] )
			+ dpR[k]*n[1];
	}

	// Energy flux
	flux[3]= Mhalf/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj))-fabs(Mhalf)/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi));

	dfdl[3*NVARS+3] = 
		dML[3]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) + Mhalf/2.0*(dci[3]*(ul[3]+pi)+ci*(1.0+dpi[3])) -
		( (Mhalf>0?1.0:-1.0)*dML[3]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi)) 
		  +fabs(Mhalf)/2.0*(-dci[3]*(ul[3]+pi)-ci*(1.0+dpi[3])) );
	dfdr[3*NVARS+3] = 
		dMR[3]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) + Mhalf/2.0*(dcj[3]*(ur[3]+pj)+cj*(1.0+dpj[3])) -
		( (Mhalf>0?1.0:-1.0)*dMR[3]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi)) 
		  +fabs(Mhalf)/2.0*(dcj[3]*(ur[3]+pj)+cj*(1.0+dpj[3])) );
	for(int k = 0; k < NVARS-1; k++)
	{
		dfdl[3*NVARS+k] = 
		  dML[k]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) +Mhalf/2.0*(dci[k]*(ul[3]+pi)+ci*dpi[k]) - 
		  ( (Mhalf>0?1.0:-1.0)*dML[k]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi)) 
			+fabs(Mhalf)/2.0*(-dci[k]*(ul[3]+pi)-ci*dpi[k]) );
		dfdr[3*NVARS+k] = 
		  dMR[k]/2.0*(ci*(ul[3]+pi)+cj*(ur[3]+pj)) +Mhalf/2.0*(dcj[k]*(ur[3]+pj)+cj*dpj[k]) - 
		  ( (Mhalf>0?1.0:-1.0)*dMR[k]/2.0*(cj*(ur[3]+pj)-ci*(ul[3]+pi)) 
			+fabs(Mhalf)/2.0*(dcj[k]*(ur[3]+pj)+cj*dpj[k]) );
	}

	for(int k = 0; k < NVARS*NVARS; k++)
		dfdl[k] = -dfdl[k];
}

AUSMPlusFlux::AUSMPlusFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{ }

void AUSMPlusFlux::get_flux(const a_real *const ul, const a_real *const ur,
		const a_real* const n, a_real *const __restrict flux)
{
	a_real ML, MR, pL, pR;
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul,pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur,pj);
	const a_real vmag2i = vxi*vxi+vyi*vyi;
	const a_real vmag2j = vxj*vxj+vyj*vyj;

	// Interface speed of sound
	a_real csi = std::sqrt((ci*ci/(g-1.0)+0.5*vmag2i)*2.0*(g-1.0)/(g+1.0));
	a_real csj = std::sqrt((cj*cj/(g-1.0)+0.5*vmag2j)*2.0*(g-1.0)/(g+1.0));
	a_real corri, corrj;
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
	const a_real chalf = (csi < csj) ? csi : csj;
	
	const a_real Mni = vni/chalf, Mnj = vnj/chalf;
	
	// split non-dimensional convection speeds (ie split Mach numbers) and split pressures
	if(std::fabs(Mni) <= 1.0)
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
	
	if(std::fabs(Mnj) <= 1.0) {
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
	const a_real Mhalf = ML+MR;
	const a_real phalf = pL+pR;

	// Fluxes
	flux[0] = chalf* (Mhalf/2.0*(ul[0]+ur[0]) -std::fabs(Mhalf)/2.0*(ur[0]-ul[0]));
	flux[1] = chalf* (Mhalf/2.0*(ul[1]+ur[1]) -std::fabs(Mhalf)/2.0*(ur[1]-ul[1])) + phalf*n[0];
	flux[2] = chalf* (Mhalf/2.0*(ul[2]+ur[2]) -std::fabs(Mhalf)/2.0*(ur[2]-ul[2])) + phalf*n[1];
	flux[3] = chalf* (Mhalf/2.0*(ul[3]+pi+ur[3]+pj) -std::fabs(Mhalf)/2.0*((ur[3]+pj)-(ul[3]+pi)));
	
	/*const a_real mplus = 0.5*(Mhalf + std::fabs(Mhalf)), mminus = 0.5*(Mhalf - std::fabs(Mhalf));
	flux[0] =  (mplus*ci*ul[0] + mminus*cj*ur[0]);
	flux[1] =  (mplus*ci*ul[1] + mminus*cj*ur[1]) + phalf*n[0];
	flux[2] =  (mplus*ci*ul[2] + mminus*cj*ur[2]) + phalf*n[1];
	flux[3] =  (mplus*ci*(ul[3]+pi) + mminus*cj*(ur[3]+pj));*/
}

void AUSMPlusFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	std::cout << " ! AUSMPlusFlux: Not implemented!\n";
}

RoeAverageBasedFlux::RoeAverageBasedFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{ }
	
void RoeAverageBasedFlux::getJacobiansRoeAveragesWrtConserved (
	const a_real ul[NVARS], const a_real ur[NVARS], const a_real n[NDIM],
	const a_real vxi, const a_real vyi, const a_real Hi,
	const a_real vxj, const a_real vyj, const a_real Hj,
	const a_real dvxi[NVARS], const a_real dvyi[NVARS], const a_real dHi[NVARS],
	const a_real dvxj[NVARS], const a_real dvyj[NVARS], const a_real dHj[NVARS],
	a_real dRiji[NVARS], a_real drhoiji[NVARS], a_real dvxiji[NVARS], a_real dvyiji[NVARS],
	a_real dvm2iji[NVARS], a_real dvniji[NVARS], a_real dHiji[NVARS], a_real dciji[NVARS],
	a_real dRijj[NVARS], a_real drhoijj[NVARS], a_real dvxijj[NVARS], a_real dvyijj[NVARS],
	a_real dvm2ijj[NVARS], a_real dvnijj[NVARS], a_real dHijj[NVARS], a_real dcijj[NVARS] )
{
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);

	// Derivatives of Rij
	dRiji[0] = 0.5/Rij * (-ur[0])/(ul[0]*ul[0]);
	dRijj[0] = 0.5/Rij / ul[0];
	for(int k = 1; k < NVARS; k++) 
	{
		dRiji[k] = 0;
		dRijj[k] = 0;
	}

	// derivatives of Roe velocities
	const a_real rden2 = (Rij+1.0)*(Rij+1.0);
	
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
		  * (((dRiji[k]*Hj+dHi[k])*(Rij+1)-(Rij*Hj+Hi)*dRiji[k])/((Rij+1)*(Rij+1)) -0.5*dvm2iji[k]);
		dcijj[k] = 0.5/cij*(g-1.0)
		  * (((dRijj[k]*Hj+Rij*dHj[k])*(Rij+1) - (Rij*Hj+Hi)*dRijj[k])/((Rij+1)*(Rij+1))
			- 0.5*dvm2ijj[k] );
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

RoeFlux::RoeFlux(const IdealGasPhysics *const analyticalflux) 
	: RoeAverageBasedFlux(analyticalflux), fixeps{1.0e-6}
{ }

void RoeFlux::get_flux(const a_real *const ul, const a_real *const ur,
		const a_real* const n, a_real *const __restrict flux)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);

	// compute Roe-averages
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);

	// eigenvalues
	a_real l[4];
	l[0] = fabs(vnij-cij); l[1] = fabs(vnij); l[2] = l[1]; l[3] = fabs(vnij+cij);
	
	// Harten entropy fix
	const a_real delta = fixeps*cij;
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		if(l[ivar] < delta)
			l[ivar] = (l[ivar]*l[ivar] + delta*delta)/(2.0*delta);
	}

	//> A_Roe * dU
	
	const a_real devn = vnj-vni, dep = pj-pi, derho = ur[0]-ul[0];
	a_real adu[NVARS];
	
	// product of eigenvalues and wave strengths
	a_real lalpha[NVARS];   
	lalpha[0] = l[0]*(dep-rhoij*cij*devn)/(2.0*cij*cij);
	lalpha[1] = l[1]*(derho - dep/(cij*cij));
	lalpha[2] = l[1]*rhoij;
	lalpha[3] = l[3]*(dep+rhoij*cij*devn)/(2.0*cij*cij);

	// un-c:
	adu[0] = lalpha[0];
	adu[1] = lalpha[0]*(vxij-cij*n[0]);
	adu[2] = lalpha[0]*(vyij-cij*n[1]);
	adu[3] = lalpha[0]*(Hij-cij*vnij);

	// un:
	adu[0] += lalpha[1];
	adu[1] += lalpha[1]*vxij +      lalpha[2]*(vxj-vxi - devn*n[0]); 
	adu[2] += lalpha[1]*vyij +      lalpha[2]*(vyj-vyi - devn*n[1]);
	adu[3] += lalpha[1]*vm2ij/2.0 + lalpha[2] *(vxij*(vxj-vxi) +vyij*(vyj-vyi) -vnij*devn);

	// un+c:
	adu[0] += lalpha[3];
	adu[1] += lalpha[3]*(vxij+cij*n[0]);
	adu[2] += lalpha[3]*(vyij+cij*n[1]);
	adu[3] += lalpha[3]*(Hij+cij*vnij);

	// get one-sided flux vectors
	a_real fi[4], fj[4];
	physics->getNormalFluxEfficiently(ul,n,vni,pi,fi);
	physics->getNormalFluxEfficiently(ur,n,vnj,pj,fj);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
		flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - adu[ivar]);
}

void RoeFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);

	// compute Roe-averages
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);
	
	//> Derivatives of the above variables
	
	a_real dpi[NVARS], dpj[NVARS], dvni[NVARS], dvnj[NVARS], dvxi[NVARS], dvxj[NVARS],
	dHi[NVARS], dHj[NVARS], dvyi[NVARS],dvyj[NVARS], 
	dRiji[NVARS], dRijj[NVARS], dvxiji[NVARS], dvyiji[NVARS], dvxijj[NVARS], dvyijj[NVARS],
	dvniji[NVARS], dvnijj[NVARS],dvm2iji[NVARS], dvm2ijj[NVARS], dciji[NVARS], dcijj[NVARS],
	drhoiji[NVARS], drhoijj[NVARS], dHiji[NVARS], dHijj[NVARS];
	for(int k = 0; k < NVARS; k++)
	{
		dpi[k] = dpj[k] = dHi[k] = dHj[k] = 0;
		dvxi[k] = dvxj[k] = dvyi[k] = dvyj[k] = dvni[k] = dvnj[k] = 0;
	}

	physics->getJacobianVarsWrtConserved(ul,n,dvxi,dvyi,dvni,dpi,dHi);
	physics->getJacobianVarsWrtConserved(ur,n,dvxj,dvyj,dvnj,dpj,dHj);

	getJacobiansRoeAveragesWrtConserved(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj,dvxi,dvyi,dHi,dvxj,dvyj,dHj,
		dRiji, drhoiji, dvxiji, dvyiji, dvm2iji, dvniji, dHiji, dciji,
		dRijj, drhoijj, dvxijj, dvyijj, dvm2ijj, dvnijj, dHijj, dcijj);

	//> eigenvalues
	
	a_real l[NVARS]; 
	l[0] = fabs(vnij-cij); l[1] = fabs(vnij); l[2] = l[1]; l[3] = fabs(vnij+cij);
	
	a_real dli[NVARS][NVARS], dlj[NVARS][NVARS];
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
	const a_real delta = fixeps*cij;
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
	
	const a_real devn = vnj-vni, dep = pj-pi, derho = ur[0]-ul[0];
	
	a_real dderhoi[NVARS], dderhoj[NVARS];
	dderhoi[0] = -1.0; dderhoj[0] = 1.0;
	for(int k = 1; k < NVARS; k++) 
	{
		dderhoi[k] = 0; 
		dderhoj[k] = 0;
	}

	// product of eigenvalues and wave strengths
	
	a_real lalpha[NVARS]; a_real dlalphai[NVARS][NVARS], dlalphaj[NVARS][NVARS];
	const a_real cij4 = cij*cij*cij*cij;

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
		dlalphai[2][k] = dli[1][NVARS]*rhoij + l[1]*drhoiji[k];
		dlalphaj[2][k] = dlj[1][NVARS]*rhoij + l[1]*drhoijj[k];
	}

	lalpha[3] = l[3]*(dep+rhoij*cij*devn)/(2.0*cij*cij);
	for(int k = 0; k < NVARS; k++)
	{
		dlalphai[3][k] = ((dli[3][k]*(dep+rhoij*cij*devn) + l[3]*(-dpi[k] +drhoiji[k]*cij*devn
			+rhoij*dciji[k]*devn+rhoij*cij*(-dvni[k])))*2.0*cij*cij - l[3]*(dep+rhoij*cij*devn)
			*4.0*cij*dciji[k]) / (4.0*cij4);
		dlalphaj[3][k] = ((dlj[3][k]*(dep+rhoij*cij*devn) + l[3]*(dpj[k] +drhoijj[k]*cij*devn
			+rhoij*dcijj[k]*devn+rhoij*cij*dvnj[k]))*2.0*cij*cij - l[3]*(dep+rhoij*cij*devn)
			*4.0*cij*dcijj[k]) / (4.0*cij4);
	}

	// dissipation terms
	
	a_real adu[NVARS]; a_real dadui[NVARS][NVARS], daduj[NVARS][NVARS];

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

		//daduj[0][k]
	}

	// un+c:
	adu[0] += lalpha[3];
	adu[1] += lalpha[3]*(vxij+cij*n[0]);
	adu[2] += lalpha[3]*(vyij+cij*n[1]);
	adu[3] += lalpha[3]*(Hij+cij*vnij);

	// get one-sided flux vectors
	a_real fi[4], fj[4];
	fi[0] = ul[0]*vni;						fj[0] = ur[0]*vnj;
	fi[1] = ul[0]*vni*vxi + pi*n[0];		fj[1] = ur[0]*vnj*vxj + pj*n[0];
	fi[2] = ul[0]*vni*vyi + pi*n[1];		fj[2] = ur[0]*vnj*vyj + pj*n[1];
	fi[3] = vni*(ul[3] + pi);				fj[3] = vnj*(ur[3] + pj);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		//flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - adu[ivar])
	}
}

HLLFlux::HLLFlux(const IdealGasPhysics *const analyticalflux) 
	: RoeAverageBasedFlux(analyticalflux)
{
}

void HLLFlux::get_flux(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, 
		const a_real* const __restrict__ n, a_real *const __restrict__ flux)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul, pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur, pj);

	//> compute Roe-averages
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);

	// Einfeldt estimate for signal speeds
	a_real sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	a_real sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const a_real sr0 = sr > 0 ? 0 : sr;
	const a_real sl0 = sl > 0 ? 0 : sl;

	// flux
	const a_real t1 = (sr0 - sl0)/(sr-sl); const a_real t2 = 1.0 - t1; 
	const a_real t3 = 0.5*(sr*fabs(sl)-sl*fabs(sr))/(sr-sl);
	flux[0] = t1*vnj*ur[0] + t2*vni*ul[0]                     - t3*(ur[0]-ul[0]);
	flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
	flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
	flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi)           - t3*(ur[3]-ul[3]);
}

/** Automatically differentiated Jacobian w.r.t. left state, 
 * generated by Tapenade 3.12 (r6213) - 13 Oct 2016 10:54.
 * Modified to remove the runtime parameter nbdirs and the change in ul. 
 * Also changed the array shape of Jacobian.
 */
void HLLFlux::getFluxJac_left(const a_real *const __restrict__ ul, 
		                      const a_real *const __restrict__ ur, 
		                      const a_real *const __restrict__ n, 
		a_real *const __restrict__ flux, a_real *const __restrict__ fluxd)
{
    a_real uld[NVARS][NVARS];
	for(int i = 0; i < NVARS; i++) {
		for(int j = 0; j < NVARS; j++)
			uld[i][j] = 0;
		uld[i][i] = 1.0;
	}
	
	const a_real g = 1.4;
    a_real Hi, Hj, ci, cj, pi, pj, vxi, vxj, vyi, vyj, vmag2i, vmag2j, vni, vnj;
    a_real Hid[NVARS], cid[NVARS], pid[NVARS], vxid[NVARS], vyid[NVARS], 
		   vmag2id[NVARS], vnid[NVARS];
    a_real fabs0;
    a_real fabs0d[NVARS];
    a_real fabs1;
    a_real fabs1d[NVARS];
    a_real arg1;
    a_real arg1d[NVARS];
    int nd;
 	a_real sld[NVARS];
    vxi = ul[1]/ul[0];
    vyi = ul[2]/ul[0];
    vmag2i = vxi*vxi + vyi*vyi;
    pi = (g-1.0)*(ul[3]-0.5*ul[0]*vmag2i);
    arg1 = g*pi/ul[0];
    for (nd = 0; nd < NVARS; ++nd) {
        vxid[nd] = (uld[1][nd]*ul[0]-ul[1]*uld[0][nd])/(ul[0]*ul[0]);
        vyid[nd] = (uld[2][nd]*ul[0]-ul[2]*uld[0][nd])/(ul[0]*ul[0]);
        vnid[nd] = n[0]*vxid[nd] + n[1]*vyid[nd];
        vmag2id[nd] = vxid[nd]*vxi + vxi*vxid[nd] + vyid[nd]*vyi + vyi*vyid[nd];
        // pressures
        pid[nd] = (g-1.0)*(uld[3][nd]-0.5*(uld[0][nd]*vmag2i+ul[0]*vmag2id[nd]));
        // speeds of sound
        arg1d[nd] = (g*pid[nd]*ul[0]-g*pi*uld[0][nd])/(ul[0]*ul[0]);
        cid[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
        // enthalpies (E + p/rho = u(3)/u(0) + p/u(0) 
		// (actually specific enthalpy := enthalpy per unit mass)
        Hid[nd] = ((uld[3][nd]+pid[nd])*ul[0]-(ul[3]+pi)*uld[0][nd])/(ul[0]*ul[0]);
        arg1d[nd] = -(ur[0]*uld[0][nd]/(ul[0]*ul[0]));
        sld[nd] = vnid[nd] - cid[nd];
    }
    vxj = ur[1]/ur[0];
    vyj = ur[2]/ur[0];
    vni = vxi*n[0] + vyi*n[1];
    vnj = vxj*n[0] + vyj*n[1];
    vmag2j = vxj*vxj + vyj*vyj;
    pj = (g-1.0)*(ur[3]-0.5*ur[0]*vmag2j);
    ci = sqrt(arg1);
    arg1 = g*pj/ur[0];
    cj = sqrt(arg1);
    Hi = (ul[3]+pi)/ul[0];
    Hj = (ur[3]+pj)/ur[0];
    // compute Roe-averages
    a_real Rij, vxij, vyij, Hij, cij, vm2ij, vnij;
    a_real Rijd[NVARS], vxijd[NVARS], vyijd[NVARS], Hijd[NVARS], cijd[NVARS], 
		   vm2ijd[NVARS], vnijd[NVARS];
    arg1 = ur[0]/ul[0];
    Rij = sqrt(arg1);
    vxij = (Rij*vxj+vxi)/(Rij+1.0);
    vyij = (Rij*vyj+vyi)/(Rij+1.0);
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
        vxijd[nd] = ((vxj*Rijd[nd]+vxid[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((vyj*Rijd[nd]+vyid[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        Hijd[nd] = ((Hj*Rijd[nd]+Hid[nd])*(Rij+1.0)-(Rij*Hj+Hi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vm2ijd[nd] = vxijd[nd]*vxij + vxij*vxijd[nd] + vyijd[nd]*vyij + vyij*vyijd[nd];
        vnijd[nd] = n[0]*vxijd[nd] + n[1]*vyijd[nd];
        arg1d[nd] = (g-1.0)*(Hijd[nd]-0.5*vm2ijd[nd]);
    }
    Hij = (Rij*Hj+Hi)/(Rij+1.0);
    vm2ij = vxij*vxij + vyij*vyij;
    vnij = vxij*n[0] + vyij*n[1];
    arg1 = (g-1.0)*(Hij-vm2ij*0.5);
    for (nd = 0; nd < NVARS; ++nd)
        cijd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
    cij = sqrt(arg1);
    // Einfeldt estimate for signal speeds
    a_real sr, sl, sr0, sl0;
    a_real srd[NVARS], sr0d[NVARS], sl0d[NVARS];
    sl = vni - ci;
    if (sl > vnij - cij) {
        for (nd = 0; nd < NVARS; ++nd)
            sld[nd] = vnijd[nd] - cijd[nd];
        sl = vnij - cij;
    }
    sr = vnj + cj;
    if (sr < vnij + cij) {
        for (nd = 0; nd < NVARS; ++nd)
            srd[nd] = vnijd[nd] + cijd[nd];
        sr = vnij + cij;
    } else
        for (nd = 0; nd < NVARS; ++nd)
            srd[nd] = 0.0;
    if (sr > 0) {
        sr0 = 0;
        for (nd = 0; nd < NVARS; ++nd)
            sr0d[nd] = 0.0;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            sr0d[nd] = srd[nd];
        sr0 = sr;
    }
    if (sl > 0) {
        sl0 = 0;
        for (nd = 0; nd < NVARS; ++nd)
            sl0d[nd] = 0.0;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            sl0d[nd] = sld[nd];
        sl0 = sl;
    }
    // flux
    a_real t1, t2, t3;
    a_real t1d[NVARS], t2d[NVARS], t3d[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        t1d[nd] = ((sr0d[nd]-sl0d[nd])*(sr-sl)-(sr0-sl0)*(srd[nd]-sld[nd]))/((sr-sl)*(sr-sl));
        t2d[nd] = -t1d[nd];
    }
    t1 = (sr0-sl0)/(sr-sl);
    t2 = 1.0 - t1;
    if (sl >= 0.0) {
        for (nd = 0; nd < NVARS; ++nd)
            fabs0d[nd] = sld[nd];
        fabs0 = sl;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fabs0d[nd] = -sld[nd];
        fabs0 = -sl;
    }
    if (sr >= 0.0) {
        for (nd = 0; nd < NVARS; ++nd)
            fabs1d[nd] = srd[nd];
        fabs1 = sr;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fabs1d[nd] = -srd[nd];
        fabs1 = -sr;
    }
    t3 = 0.5*(sr*fabs0-sl*fabs1)/(sr-sl);
    for (nd = 0; nd < NVARS; ++nd) {
        t3d[nd] = (0.5*(srd[nd]*fabs0+sr*fabs0d[nd]-sld[nd]*fabs1-sl*fabs1d[nd])*(sr-sl)
				-0.5*(sr*fabs0-sl*fabs1)*(srd[nd]-sld[nd]))/((sr-sl)*(sr-sl));
        fluxd[0*NVARS+nd] = vnj*ur[0]*t1d[nd] + (t2d[nd]*vni+t2*vnid[nd])*ul[0] 
			+ t2*vni*uld[0][nd] - t3d[nd]*(ur[0]-ul[0]) + t3*uld[0][nd];
    }
    flux[0] = t1*vnj*ur[0] + t2*vni*ul[0] - t3*(ur[0]-ul[0]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[1*NVARS+nd] = (vnj*ur[1]+pj*n[0])*t1d[nd] + t2d[nd]*(vni*ul[1]+pi*n[0]) 
			+ t2*(vnid[nd]*ul[1]+vni*uld[1][nd]+n[0]*pid[nd]) - t3d[nd]*(ur[1]-ul[1]) 
			+ t3*uld[1][nd];
    flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[2*NVARS+nd] = (vnj*ur[2]+pj*n[1])*t1d[nd] + t2d[nd]*(vni*ul[2]+pi*n[1]) 
			+ t2*(vnid[nd]*ul[2]+vni*uld[2][nd]+n[1]*pid[nd]) - t3d[nd]*(ur[2]-ul[2]) 
			+ t3*uld[2][nd];
    flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[3*NVARS+nd] = vnj*ur[0]*Hj*t1d[nd] + (t2d[nd]*vni+t2*vnid[nd])*ul[0]*Hi 
			+ t2*vni*(uld[0][nd]*Hi+ul[0]*Hid[nd]) - t3d[nd]*(ur[3]-ul[3]) + t3*uld[3][nd];
    flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi) - t3*(ur[3]-ul[3]);
}

/** Automatically differentiated Jacobian w.r.t. right state, 
 * generated by Tapenade 3.12 (r6213) - 13 Oct 2016 10:54.
 * Modified to remove the runtime parameter nbdirs and the differential of ul. 
 * Also changed the array shape of Jacobian.
 */
void HLLFlux::getFluxJac_right(const a_real *const ul, const a_real *const ur, 
		const a_real *const n, 
		a_real *const __restrict flux, a_real *const __restrict fluxd)
{
    a_real urd[NVARS][NVARS];
	for(int i = 0; i < NVARS; i++) {
		for(int j = 0; j < NVARS; j++)
			urd[i][j] = 0;
		urd[i][i] = 1.0;
	}

    a_real Hi, Hj, ci, cj, pi, pj, vxi, vxj, vyi, vyj, vmag2i, vmag2j, vni, vnj;
    a_real Hjd[NVARS], cjd[NVARS], pjd[NVARS], vxjd[NVARS], vyjd[NVARS], 
		   vmag2jd[NVARS], vnjd[NVARS];
    a_real fabs0;
    a_real fabs0d[NVARS];
    a_real fabs1;
    a_real fabs1d[NVARS];
    a_real arg1;
    a_real arg1d[NVARS];
    int nd;
    vxi = ul[1]/ul[0];
    vyi = ul[2]/ul[0];
    vxj = ur[1]/ur[0];
    vyj = ur[2]/ur[0];
    vmag2i = vxi*vxi + vyi*vyi;
    vmag2j = vxj*vxj + vyj*vyj;
    // pressures
    pi = (g-1.0)*(ul[3]-0.5*ul[0]*vmag2i);
    pj = (g-1.0)*(ur[3]-0.5*ur[0]*vmag2j);
    // speeds of sound
    arg1 = g*pi/ul[0];
    ci = sqrt(arg1);
    arg1 = g*pj/ur[0];
    for (nd = 0; nd < NVARS; ++nd) {
        vxjd[nd] = (urd[1][nd]*ur[0]-ur[1]*urd[0][nd])/(ur[0]*ur[0]);
        vyjd[nd] = (urd[2][nd]*ur[0]-ur[2]*urd[0][nd])/(ur[0]*ur[0]);
        vnjd[nd] = n[0]*vxjd[nd] + n[1]*vyjd[nd];
        vmag2jd[nd] = vxjd[nd]*vxj + vxj*vxjd[nd] + vyjd[nd]*vyj + vyj*vyjd[nd];
        pjd[nd] = (g-1.0)*(urd[3][nd]-0.5*(urd[0][nd]*vmag2j+ur[0]*vmag2jd[nd]));
        arg1d[nd] = (g*pjd[nd]*ur[0]-g*pj*urd[0][nd])/(ur[0]*ur[0]);
        cjd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
        Hjd[nd] = ((urd[3][nd]+pjd[nd])*ur[0]-(ur[3]+pj)*urd[0][nd])/(ur[0]*ur[0]);
        arg1d[nd] = urd[0][nd]/ul[0];
    }
    vni = vxi*n[0] + vyi*n[1];
    vnj = vxj*n[0] + vyj*n[1];
    cj = sqrt(arg1);
    // enthalpies (E + p/rho = u(3)/u(0) + p/u(0) 
	// (actually specific enthalpy := enthalpy per unit mass)
    Hi = (ul[3]+pi)/ul[0];
    Hj = (ur[3]+pj)/ur[0];
    // compute Roe-averages
    a_real Rij, vxij, vyij, Hij, cij, vm2ij, vnij;
    a_real Rijd[NVARS], vxijd[NVARS], vyijd[NVARS], Hijd[NVARS], cijd[NVARS], 
		   vm2ijd[NVARS], vnijd[NVARS];
    arg1 = ur[0]/ul[0];
    Rij = sqrt(arg1);
    vxij = (Rij*vxj+vxi)/(Rij+1.0);
    vyij = (Rij*vyj+vyi)/(Rij+1.0);
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
        vxijd[nd] = ((Rijd[nd]*vxj+Rij*vxjd[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((Rijd[nd]*vyj+Rij*vyjd[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        Hijd[nd] = ((Rijd[nd]*Hj+Rij*Hjd[nd])*(Rij+1.0)-(Rij*Hj+Hi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vm2ijd[nd] = vxijd[nd]*vxij + vxij*vxijd[nd] + vyijd[nd]*vyij + vyij*vyijd[nd];
        vnijd[nd] = n[0]*vxijd[nd] + n[1]*vyijd[nd];
        arg1d[nd] = (g-1.0)*(Hijd[nd]-0.5*vm2ijd[nd]);
    }
    Hij = (Rij*Hj+Hi)/(Rij+1.0);
    vm2ij = vxij*vxij + vyij*vyij;
    vnij = vxij*n[0] + vyij*n[1];
    arg1 = (g-1.0)*(Hij-vm2ij*0.5);
    for (nd = 0; nd < NVARS; ++nd)
        cijd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
    cij = sqrt(arg1);
    // Einfeldt estimate for signal speeds
    a_real sr, sl, sr0, sl0;
    a_real srd[NVARS], sld[NVARS], sr0d[NVARS], sl0d[NVARS];
    sl = vni - ci;
    if (sl > vnij - cij) {
        for (nd = 0; nd < NVARS; ++nd)
            sld[nd] = vnijd[nd] - cijd[nd];
        sl = vnij - cij;
    } else
        for (nd = 0; nd < NVARS; ++nd)
            sld[nd] = 0.0;
    for (nd = 0; nd < NVARS; ++nd)
        srd[nd] = vnjd[nd] + cjd[nd];
    sr = vnj + cj;
    if (sr < vnij + cij) {
        for (nd = 0; nd < NVARS; ++nd)
            srd[nd] = vnijd[nd] + cijd[nd];
        sr = vnij + cij;
    }
    if (sr > 0) {
        sr0 = 0;
        for (nd = 0; nd < NVARS; ++nd)
            sr0d[nd] = 0.0;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            sr0d[nd] = srd[nd];
        sr0 = sr;
    }
    if (sl > 0) {
        sl0 = 0;
        for (nd = 0; nd < NVARS; ++nd)
            sl0d[nd] = 0.0;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            sl0d[nd] = sld[nd];
        sl0 = sl;
    }
    // flux
    a_real t1, t2, t3;
    a_real t1d[NVARS], t2d[NVARS], t3d[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        t1d[nd] = ((sr0d[nd]-sl0d[nd])*(sr-sl)-(sr0-sl0)*(srd[nd]-sld[nd]))/((sr-sl)*(sr-sl));
        t2d[nd] = -t1d[nd];
    }
    t1 = (sr0-sl0)/(sr-sl);
    t2 = 1.0 - t1;
    if (sl >= 0.0) {
        for (nd = 0; nd < NVARS; ++nd)
            fabs0d[nd] = sld[nd];
        fabs0 = sl;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fabs0d[nd] = -sld[nd];
        fabs0 = -sl;
    }
    if (sr >= 0.0) {
        for (nd = 0; nd < NVARS; ++nd)
            fabs1d[nd] = srd[nd];
        fabs1 = sr;
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fabs1d[nd] = -srd[nd];
        fabs1 = -sr;
    }
    t3 = 0.5*(sr*fabs0-sl*fabs1)/(sr-sl);
    for (nd = 0; nd < NVARS; ++nd) {
        t3d[nd] = (0.5*(srd[nd]*fabs0+sr*fabs0d[nd]-sld[nd]*fabs1-sl*fabs1d[nd])*(sr-sl)
			-0.5*(sr*fabs0-sl*fabs1)*(srd[nd]-sld[nd]))/((sr-sl)*(sr-sl));
        fluxd[0*NVARS+nd] = (t1d[nd]*vnj+t1*vnjd[nd])*ur[0] + t1*vnj*urd[0][nd] + vni*ul[0]*t2d[nd] 
			- t3d[nd]*(ur[0]-ul[0]) - t3*urd[0][nd];
    }
    flux[0] = t1*vnj*ur[0] + t2*vni*ul[0] - t3*(ur[0]-ul[0]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[1*NVARS+nd] = t1d[nd]*(vnj*ur[1]+pj*n[0]) 
			+ t1*(vnjd[nd]*ur[1]+vnj*urd[1][nd]+n[0]*pjd[nd]) 
			+ (vni*ul[1]+pi*n[0])*t2d[nd] - t3d[nd]*(ur[1]-ul[1]) - t3*urd[1][nd];
    flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[2*NVARS+nd] = t1d[nd]*(vnj*ur[2]+pj*n[1]) 
			+ t1*(vnjd[nd]*ur[2]+vnj*urd[2][nd]+n[1]*pjd[nd]) 
			+ (vni*ul[2]+pi*n[1])*t2d[nd] - t3d[nd]*(ur[2]-ul[2]) - t3*urd[2][nd];
    flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[3*NVARS+nd] = (t1d[nd]*vnj+t1*vnjd[nd])*ur[0]*Hj 
			+ t1*vnj*(urd[0][nd]*Hj+ur[0]*Hjd[nd]) 
			+ vni*ul[0]*Hi*t2d[nd] - t3d[nd]*(ur[3]-ul[3]) - t3*urd[3][nd];
    flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi) - t3*(ur[3]-ul[3]);
}

void HLLFlux::get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	a_real flux[NVARS];
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

void HLLFlux::get_flux_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, 
		a_real *const __restrict flux, a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

/** The linearization assumes `locally frozen' signal speeds. 
 * According to Batten, Lechziner and Goldberg, this should be fine.
 */
void HLLFlux::get_frozen_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	const a_real vxi = ul[1]/ul[0]; const a_real vyi = ul[2]/ul[0];
	const a_real vxj = ur[1]/ur[0]; const a_real vyj = ur[2]/ur[0];
	const a_real vni = vxi*n[0] + vyi*n[1];
	const a_real vnj = vxj*n[0] + vyj*n[1];
	const a_real vmag2i = vxi*vxi + vyi*vyi;
	const a_real vmag2j = vxj*vxj + vyj*vyj;
	// pressures
	const a_real pi = (g-1.0)*(ul[3] - 0.5*ul[0]*vmag2i);
	const a_real pj = (g-1.0)*(ur[3] - 0.5*ur[0]*vmag2j);
	// speeds of sound
	const a_real ci = sqrt(g*pi/ul[0]);
	const a_real cj = sqrt(g*pj/ur[0]);
	// enthalpies (E + p/rho = u(3)/u(0) + p/u(0) 
	// (actually specific enthalpy := enthalpy per unit mass)
	const a_real Hi = (ul[3] + pi)/ul[0];
	const a_real Hj = (ur[3] + pj)/ur[0];

	// compute Roe-averages
	const a_real Rij = sqrt(ur[0]/ul[0]);
	const a_real vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	const a_real vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	const a_real Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	const a_real vm2ij = vxij*vxij + vyij*vyij;
	const a_real vnij = vxij*n[0] + vyij*n[1];
	const a_real cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	// Einfeldt estimate for signal speeds
	a_real sr, sl;
	sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const a_real sr0 = sr > 0 ? 0 : sr;
	const a_real sl0 = sl > 0 ? 0 : sl;
	const a_real t1 = (sr0 - sl0)/(sr-sl); 
	const a_real t2 = 1.0 - t1; 
	const a_real t3 = 0.5*(sr*fabs(sl)-sl*fabs(sr))/(sr-sl);
	
	// get flux jacobians
	physics->getJacobianNormalFluxWrtConserved(ul, n, dfdl);
	physics->getJacobianNormalFluxWrtConserved(ur, n, dfdr);
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

HLLCFlux::HLLCFlux(const IdealGasPhysics *const analyticalflux) 
	: RoeAverageBasedFlux(analyticalflux)
{
}

void HLLCFlux::get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
		a_real *const __restrict flux)
{
	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul, pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur, pj);

	// compute Roe-averages
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);

	// estimate signal speeds
	a_real sr, sl;
	sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	const a_real sm = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj ) 
		/ ( ur[0]*(sr-vnj) - ul[0]*(sl-vni) );

	// compute fluxes
	
	if(sl > 0)
		physics->getNormalFluxEfficiently(ul,n,vni,pi,flux);

	else if(sl <= 0 && sm > 0)
	{
		physics->getNormalFluxEfficiently(ul,n,vni,pi,flux);

		const a_real pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
		a_real utemp[NVARS];
		utemp[0] = ul[0] * (sl - vni)/(sl-sm);
		utemp[1] = ( (sl-vni)*ul[1] + (pstar-pi)*n[0] )/(sl-sm);
		utemp[2] = ( (sl-vni)*ul[2] + (pstar-pi)*n[1] )/(sl-sm);
		utemp[3] = ( (sl-vni)*ul[3] - pi*vni + pstar*sm )/(sl-sm);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sl * ( utemp[ivar] - ul[ivar]);
	}
	else if(sm <= 0 && sr >= 0)
	{
		physics->getNormalFluxEfficiently(ur,n,vnj,pj,flux);

		const a_real pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
		a_real utemp[NVARS];
		utemp[0] = ur[0] * (sr - vnj)/(sr-sm);
		utemp[1] = ( (sr-vnj)*ur[1] + (pstar-pj)*n[0] )/(sr-sm);
		utemp[2] = ( (sr-vnj)*ur[2] + (pstar-pj)*n[1] )/(sr-sm);
		utemp[3] = ( (sr-vnj)*ur[3] - pj*vnj + pstar*sm )/(sr-sm);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sr * ( utemp[ivar] - ur[ivar]);
	}
	else
		physics->getNormalFluxEfficiently(ur,n,vnj,pj,flux);
}

void HLLCFlux::get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	a_real flux[NVARS];

	a_real vxi, vxj, vyi, vyj, vni, vnj, pi, pj, Hi, Hj, ci, cj;
	physics->getVarsFromConserved(ul, n, vxi, vyi, vni, pi, Hi);
	physics->getVarsFromConserved(ur, n, vxj, vyj, vnj, pj, Hj);
	ci = physics->getSoundSpeedFromConservedEfficiently(ul, pi);
	cj = physics->getSoundSpeedFromConservedEfficiently(ur, pj);

	// compute Roe-averages
	a_real Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij;	
	getRoeAverages(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj, Rij,rhoij,vxij,vyij,vm2ij,vnij,Hij,cij);
	
	//> Derivatives of the above variables
	
	a_real dpi[NVARS], dpj[NVARS], dvni[NVARS], dvnj[NVARS], dvxi[NVARS], dvxj[NVARS],
	dHi[NVARS], dHj[NVARS], dvyi[NVARS],dvyj[NVARS], dci[NVARS], dcj[NVARS], 
	dRiji[NVARS], dRijj[NVARS], dvxiji[NVARS], dvyiji[NVARS], dvxijj[NVARS], dvyijj[NVARS],
	dvniji[NVARS], dvnijj[NVARS],dvm2iji[NVARS], dvm2ijj[NVARS], dciji[NVARS], dcijj[NVARS],
	drhoiji[NVARS], drhoijj[NVARS], dHiji[NVARS], dHijj[NVARS];
	for(int k = 0; k < NVARS; k++)
	{
		dpi[k] = dpj[k] = dHi[k] = dHj[k] = dci[k] = dcj[k] = 0;
		dvxi[k] = dvxj[k] = dvyi[k] = dvyj[k] = dvni[k] = dvnj[k] = 0;
	}

	physics->getJacobianVarsWrtConserved(ul,n,dvxi,dvyi,dvni,dpi,dHi);
	physics->getJacobianVarsWrtConserved(ur,n,dvxj,dvyj,dvnj,dpj,dHj);
	physics->getJacobianSoundSpeedWrtConservedEfficiently(ul,pi,dpi,ci,dci);
	physics->getJacobianSoundSpeedWrtConservedEfficiently(ur,pj,dpj,cj,dcj);

	getJacobiansRoeAveragesWrtConserved(ul,ur,n,vxi,vyi,Hi,vxj,vyj,Hj,dvxi,dvyi,dHi,dvxj,dvyj,dHj,
		dRiji, drhoiji, dvxiji, dvyiji, dvm2iji, dvniji, dHiji, dciji,
		dRijj, drhoijj, dvxijj, dvyijj, dvm2ijj, dvnijj, dHijj, dcijj);

	// estimate signal speeds 
	a_real sr, sl, dsli[NVARS], dslj[NVARS], dsri[NVARS], dsrj[NVARS];
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
		dsrj[k] = dvnj[k] - dcj[k];
	}
	if(sr < vnij+cij) {
		sr = vnij+cij;
		for(int k = 0; k < NVARS; k++) {
			dsri[k] = dvniji[k] + dciji[k];
			dsrj[k] = dvnijj[k] + dcijj[k];
		}
	}

	const a_real num = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj );
	const a_real denom = (ur[0]*(sr-vnj) - ul[0]*(sl-vni));

	const a_real sm = num / denom;
	a_real dsmi[NVARS], dsmj[NVARS];

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
		physics->getNormalFluxEfficiently(ul,n,vni,pi,flux);

		physics->getJacobianNormalFluxWrtConserved(ul,n,dfdl);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdr[k] = 0;
	}
	else if(sl <= 0 && sm > 0)
	{
		physics->getNormalFluxEfficiently(ul,n,vni,pi,flux);

		physics->getJacobianNormalFluxWrtConserved(ul,n,dfdl);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdr[k] = 0;

		const a_real pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
		
		a_real dpsi[NVARS], dpsj[NVARS];
		dpsi[0] = (vni-sl)*(vni-sm) +ul[0]*(dvni[0]-dsli[0])*(vni-sm)
			+ul[0]*(vni-sl)*(dvni[0]-dsmi[0]) + dpi[0];
		dpsj[0] = ul[0]*((-dslj[0])*(vni-sm) + (vni-sl)*(-dsmj[0]));
		for(int k = 1; k < NVARS; k++) 
		{
			dpsi[k] = ul[0]*(dvni[k]-dsli[k])*(vni-sm)+(vni-sl)*(dvni[k]-dsmi[k]) + dpi[k];
			dpsj[k] = ul[0]*((-dslj[k])*(vni-sm)+(vni-sl)*(-dsmj[k]));
		}
		
		a_real utemp[NVARS], dutempi[NVARS][NVARS], dutempj[NVARS][NVARS];

		utemp[0] = ul[0] * (sl - vni)/(sl-sm);

		dutempi[0][0]=ul[0]*((dsli[0]-dvni[0])*(sl-sm)-(sl-vni)*(dsli[0]-dsmi[0]))/((sl-sm)*(sl-sm))
			+ (sl-vni)/(sl-sm);
		dutempj[0][0]=ul[0]*(dslj[0]*(sl-sm)-(sl-vni)*(dslj[0]-dsmj[0])) / ((sl-sm)*(sl-sm));
		for(int k = 1; k < NVARS; k++) 
		{
			dutempi[0][k]=ul[0]*((dsli[k]-dvni[k])*(sl-sm)-(sl-vni)*(dsli[k]-dsmi[k]))
				/ ((sl-sm)*(sl-sm));
			dutempj[0][k]=ul[0]*(dslj[k]*(sl-sm)-(sl-vni)*(dslj[k]-dsmj[k])) / ((sl-sm)*(sl-sm));
		}

		utemp[1] = ( (sl-vni)*ul[1] + (pstar-pi)*n[0] )/(sl-sm);

		for(int k = 0; k < NVARS; k++)
		{
			if(k == 1) continue;
			dutempi[1][k]= ( ((dsli[k]-dvni[k])*ul[1] + (dpsi[k]-dpi[k])*n[0])*(sl-sm)
				- ((sl-vni)*ul[1]+(pstar-pi)*n[0])*(dsli[k]-dsmi[k]) )/((sl-sm)*(sl-sm));
			dutempj[1][k]= ( (dslj[k]*ul[1] + dpsj[k]*n[0])*(sl-sm) 
				- ((sl-vni)*ul[1]+(pstar-pi)*n[0])*(dslj[k]-dsmj[k]) )/((sl-sm)*(sl-sm));
		}
		dutempi[1][1]= ( ((dsli[1]-dvni[1])*ul[1]+(sl-vni) + (dpsi[1]-dpi[1])*n[0])*(sl-sm)
				- ((sl-vni)*ul[1]+(pstar-pi)*n[0])*(dsli[1]-dsmi[1]) )/((sl-sm)*(sl-sm));
		dutempj[1][1]= ( (dslj[1]*ul[1] + dpsj[1]*n[0])*(sl-sm) 
			- ((sl-vni)*ul[1]+(pstar-pi)*n[0])*(dslj[1]-dsmj[1]) )/((sl-sm)*(sl-sm));

		utemp[2] = ( (sl-vni)*ul[2] + (pstar-pi)*n[1] )/(sl-sm);

		for(int k = 0; k < NVARS; k++)
		{
			if(k == 2) continue;
			dutempi[2][k]= ( ((dsli[k]-dvni[k])*ul[2] + (dpsi[k]-dpi[k])*n[1])*(sl-sm)
				- ((sl-vni)*ul[2]+(pstar-pi)*n[1])*(dsli[k]-dsmi[k]) )/((sl-sm)*(sl-sm));
			dutempj[2][k]= ( (dslj[k]*ul[2] + dpsj[k]*n[1])*(sl-sm) 
				- ((sl-vni)*ul[2]+(pstar-pi)*n[1])*(dslj[k]-dsmj[k]) )/((sl-sm)*(sl-sm));
		}
		dutempi[2][2]= ( ((dsli[2]-dvni[2])*ul[2]+(sl-vni) + (dpsi[2]-dpi[2])*n[1])*(sl-sm)
				- ((sl-vni)*ul[2]+(pstar-pi)*n[1])*(dsli[2]-dsmi[2]) )/((sl-sm)*(sl-sm));
		dutempj[2][2]= ( (dslj[2]*ul[2] + dpsj[2]*n[1])*(sl-sm) 
			- ((sl-vni)*ul[2]+(pstar-pi)*n[1])*(dslj[2]-dsmj[2]) )/((sl-sm)*(sl-sm));

		utemp[3] = ( (sl-vni)*ul[3] - pi*vni + pstar*sm )/(sl-sm);

		for(int k = 0; k < NVARS-1; k++) {
			dutempi[3][k]= ( ((dsli[k]-dvni[k])*ul[3] -dpi[k]*vni-pi*dvni[k] 
				+dpsi[k]*sm+pstar*dsmi[k]) * (sl-sm) 
				- ((sl-vni)*ul[3]-pi*vni+pstar*sm)*(dsli[k]-dsmi[k]) )/((sl-sm)*(sl-sm));
			dutempj[3][k]= ( (dslj[k]*ul[3] + dpsj[k]*sm+pstar*dsmj[k])*(sl-sm)
				- ((sl-vni)*ul[3]-pi*vni+pstar*sm)*(dslj[k]-dsmj[k]) )/((sl-sm)*(sl-sm));
		}
		dutempi[3][3]= ( ((dsli[3]-dvni[3])*ul[3]+(sl-vni) -dpi[3]*vni-pi*dvni[3] 
			+dpsi[3]*sm+pstar*dsmi[3]) * (sl-sm) 
			- ((sl-vni)*ul[3]-pi*vni+pstar*sm)*(dsli[3]-dsmi[3]) )/((sl-sm)*(sl-sm));
		dutempj[3][3]= ( (dslj[3]*ul[3] + dpsj[3]*sm+pstar*dsmj[3])*(sl-sm)
			- ((sl-vni)*ul[3]-pi*vni+pstar*sm)*(dslj[3]-dsmj[3]) )/((sl-sm)*(sl-sm));

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			flux[ivar] += sl * ( utemp[ivar] - ul[ivar]);
			for(int k = 0; k < NVARS; k++)
			{
				dfdl[ivar*NVARS+k] += dsli[k]*(utemp[ivar]-ul[ivar]) 
					+ sl*(dutempi[ivar][k] - (ivar==k ? 1.0 : 0.0));
				dfdr[ivar*NVARS+k] += dslj[k]*(utemp[ivar]-ul[ivar]) + sl*dutempj[ivar][k];
			}
		}
	}
	else if(sm <= 0 && sr >= 0)
	{
		physics->getNormalFluxEfficiently(ur,n,vnj,pj,flux);

		physics->getJacobianNormalFluxWrtConserved(ur,n,dfdr);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdl[k] = 0;

		const a_real pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;

		a_real dpsi[NVARS], dpsj[NVARS];
		for(int k = 1; k < NVARS; k++)
		{
			dpsi[k] = ur[0]*(-dsri[k]*(vnj-sm)+(vnj-sr)*(-dsmi[k]));
			dpsj[k] = ur[0]*((dvnj[k]-dsrj[k])*(vnj-sm)+(vnj-sr)*(dvnj[k]-dsmj[k])) + dpj[k];
		}
		dpsi[0] = ur[0]*(-dsri[0]*(vnj-sm)+(vnj-sr)*(-dsmi[0]));
		dpsj[0] = (vnj-sr)*(vnj-sm) +ur[0]*(dvnj[0]-dsrj[0])*(vnj-sm) 
			+ur[0]*(vnj-sr)*(dvnj[0]-dsmj[0]) + dpj[0];
		
		a_real utemp[NVARS];
		a_real dutempi[NVARS][NVARS], dutempj[NVARS][NVARS];

		utemp[0] = ur[0] * (sr - vnj)/(sr-sm);

		for(int k = 1; k < NVARS; k++)
		{
			dutempi[0][k] = ur[0]*(dsri[k]*(sr-sm)-(sr-vnj)*(dsri[k]-dsmi[k])) / ((sr-sm)*(sr-sm));
			dutempj[0][k] = ur[0]*((dsrj[k]-dvnj[k])*(sr-sm)-(sr-vnj)*(dsrj[k]-dsmj[k])) / 
				((sr-sm)*(sr-sm));
		}
		dutempi[0][0] = ur[0]*(dsri[0]*(sr-sm)-(sr-vnj)*(dsri[0]-dsmi[0])) / ((sr-sm)*(sr-sm));
		dutempj[0][0] = ur[0]*((dsrj[0]-dvnj[0])*(sr-sm)-(sr-vnj)*(dsrj[0]-dsmj[0])) / 
			((sr-sm)*(sr-sm)) + (sr-vnj)/(sr-sm);

		utemp[1] = ( (sr-vnj)*ur[1] + (pstar-pj)*n[0] )/(sr-sm);

		for(int k = 0; k < NVARS; k++)
		{
			if(k == 1) continue;
			dutempi[1][k] = ((dsri[k]*ur[1] +dpsi[k]*n[0])*(sr-sm) 
				-((sr-vnj)*ur[1]+(pstar-pj)*n[0])*(dsri[k]-dsmi[k])) / ((sr-sm)*(sr-sm));
			dutempj[1][k] = (((dsrj[k]-dvnj[k])*ur[1] +(dpsj[k]-dpj[k])*n[0])*(sr-sm) -
				((sr-vnj)*ur[1]+(pstar-pj)*n[0])*(dsrj[k]-dsmj[k])) / ((sr-sm)*(sr-sm));
		}
		dutempi[1][1] = ((dsri[1]*ur[1] +dpsi[1]*n[0])*(sr-sm) +((sr-vnj)*ur[1]+(pstar-pj)*n[0])*
			(dsri[1]-dsmi[1])) / ((sr-sm)*(sr-sm));
		dutempj[1][1] = (((dsrj[1]-dvnj[1])*ur[1] +sr-vnj +(dpsj[1]-dpj[1])*n[0])*(sr-sm) -
			((sr-vnj)*ur[1]+(pstar-pj)*n[0])*(dsrj[1]-dsmj[1])) / ((sr-sm)*(sr-sm));

		utemp[2] = ( (sr-vnj)*ur[2] + (pstar-pj)*n[1] )/(sr-sm);

		for(int k = 0; k < NVARS; k++)
		{
			if(k == 2) continue;
			dutempi[2][k] = ((dsri[k]*ur[2] +dpsi[k]*n[1])*(sr-sm) 
				-((sr-vnj)*ur[2]+(pstar-pj)*n[1])*(dsri[k]-dsmi[k])) / ((sr-sm)*(sr-sm));
			dutempj[2][k] = (((dsrj[k]-dvnj[k])*ur[2] +(dpsj[k]-dpj[k])*n[1])*(sr-sm) -
				((sr-vnj)*ur[2]+(pstar-pj)*n[1])*(dsrj[k]-dsmj[k])) / ((sr-sm)*(sr-sm));
		}
		dutempi[2][2] = ((dsri[2]*ur[2] +dpsi[2]*n[1])*(sr-sm) -((sr-vnj)*ur[2]+(pstar-pj)*n[1])*
			(dsri[2]-dsmi[2])) / ((sr-sm)*(sr-sm));
		dutempj[2][2] = (((dsrj[2]-dvnj[2])*ur[2] +sr-vnj +(dpsj[2]-dpj[2])*n[1])*(sr-sm) -
			((sr-vnj)*ur[2]+(pstar-pj)*n[1])*(dsrj[2]-dsmj[2])) / ((sr-sm)*(sr-sm));

		utemp[3] = ( (sr-vnj)*ur[3] - pj*vnj + pstar*sm )/(sr-sm);

		for(int k = 0; k < NVARS-1; k++)
		{
			dutempi[3][k] = ((dsri[k]*ur[3] +dpsi[k]*sm+pstar*dsmi[k])*(sr-sm)
				-((sr-vnj)*ur[3]-pj*vnj+pstar*sm)*(dsri[k]-dsmi[k])) / ((sr-sm)*(sr-sm));
			dutempj[3][k]= (((dsrj[k]-dvnj[k])*ur[3] -dpj[k]*vnj-pj*dvnj[k] 
				+dpsj[k]*sm+pstar*dsmj[k])*(sr-sm) 
				-((sr-vnj)*ur[3]-pj*vnj+pstar*sm)*(dsrj[k]-dsmj[k])) / ((sr-sm)*(sr-sm));
		}
		dutempi[3][3] = ((dsri[3]*ur[3] +dpsi[3]*sm+pstar*dsmi[3])*(sr-sm)
			-((sr-vnj)*ur[3]-pj*vnj+pstar*sm)*(dsri[3]-dsmi[3])) / ((sr-sm)*(sr-sm));
		dutempj[3][3]= (((dsrj[3]-dvnj[3])*ur[3]+(sr-vnj) -dpj[3]*vnj-pj*dvnj[3] 
			+dpsj[3]*sm+pstar*dsmj[3])*(sr-sm) 
			-((sr-vnj)*ur[3]-pj*vnj+pstar*sm)*(dsrj[3]-dsmj[3])) / ((sr-sm)*(sr-sm));

		for(int ivar = 0; ivar < NVARS; ivar++)
		{
			flux[ivar] += sr * ( utemp[ivar] - ur[ivar]);

			for(int k = 0; k < NVARS; k++) {
				dfdl[ivar*NVARS+k] += dsri[k]*(utemp[ivar]-ur[ivar]) +sr*dutempi[ivar][k];
				dfdr[ivar*NVARS+k] += dsrj[k]*(utemp[ivar]-ur[ivar])
										+ sr*(dutempj[ivar][k] - (ivar==k ? 1.0:0.0));
			}
		}
	}
	else
	{
		physics->getNormalFluxEfficiently(ur,n,vnj,pj,flux);

		physics->getJacobianNormalFluxWrtConserved(ur,n,dfdr);
		for(int k = 0; k < NVARS*NVARS; k++)
			dfdl[k] = 0;
	}

	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

void HLLCFlux::get_flux_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, 
		a_real *const __restrict flux, a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	std::cout << " !!!! HLLC Jacobian not available!!\n";
}

} // end namespace acfd

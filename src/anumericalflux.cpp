/** \file anumericalflux.cpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

/* Tapenade notes:
 * No consts
 * #defines work but get replaced
 */

#include "anumericalflux.hpp"

namespace acfd {

InviscidFlux::InviscidFlux(const IdealGasPhysics *const phyctx) 
	: physics(phyctx), g(phyctx->g)
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

void LocalLaxFriedrichsFlux::get_flux(const a_real *const __restrict__ ul, 
		const a_real *const __restrict__ ur, 
		const a_real* const __restrict__ n, 
		a_real *const __restrict__ flux)
{
	//a_real vni, vnj, pi, pj, ci, cj, eig;

	//calculate presures from u
	const a_real pi = (g-1)*(ul[3] - 0.5*(std::pow(ul[1],2)+std::pow(ul[2],2))/ul[0]);
	const a_real pj = (g-1)*(ur[3] - 0.5*(std::pow(ur[1],2)+std::pow(ur[2],2))/ur[0]);
	//calculate speeds of sound
	const a_real ci = std::sqrt(g*pi/ul[0]);
	const a_real cj = std::sqrt(g*pj/ur[0]);
	//calculate normal velocities
	const a_real vni = (ul[1]*n[0] + ul[2]*n[1])/ul[0];
	const a_real vnj = (ur[1]*n[0] + ur[2]*n[1])/ur[0];
	// max eigenvalue
	/*a_real vmagl = std::sqrt(ul[1]*ul[1]+ul[2]*ul[2])/ul[0];
	a_real vmagr = std::sqrt(ur[1]*ur[1]+ur[2]*ur[2])/ur[0];
	eig = vmagl+ci > vmagr+cj ? vmagl+ci : vmagr+cj;*/
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
	a_real eig;

	//calculate presures from u
	const a_real pi = (g-1)*(ul[3] - 0.5*(pow(ul[1],2)+pow(ul[2],2))/ul[0]);
	const a_real pj = (g-1)*(ur[3] - 0.5*(pow(ur[1],2)+pow(ur[2],2))/ur[0]);
	//calculate speeds of sound
	const a_real ci = sqrt(g*pi/ul[0]);
	const a_real cj = sqrt(g*pj/ur[0]);
	//calculate normal velocities
	const a_real vni = (ul[1]*n[0] + ul[2]*n[1])/ul[0];
	const a_real vnj = (ur[1]*n[0] + ur[2]*n[1])/ur[0];
	
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
	physics->evaluate_normal_jacobian(ul, n, dfdl);
	physics->evaluate_normal_jacobian(ur, n, dfdr);

	// add contributions to left derivative
	for(int i = 0; i < NVARS; i++)
	{
		dfdl[i*NVARS+i] -= -eig;
		/*for(int j = 0; j < NVARS; j++)
			dfdl[i*NVARS+j] -= dedu[j]*(ur[i]-ul[i]);*/
	}

	// add contributions to right derivarive
	for(int i = 0; i < NVARS; i++)
	{
		dfdr[i*NVARS+i] -= eig;
		/*for(int j = 0; j < NVARS; j++)
			dfdr[i*NVARS+j] -= dedu[j]*(ur[i]-ul[i]);*/
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

// full linearization, inspite of the name; no better than frozen version
void LocalLaxFriedrichsFlux::get_frozen_jacobian(const a_real *const ul, 
		const a_real *const ur,
		const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	a_real eig; 

	//calculate presures from u
	const a_real pi = (g-1)*(ul[3] - 0.5*(pow(ul[1],2)+pow(ul[2],2))/ul[0]);
	const a_real pj = (g-1)*(ur[3] - 0.5*(pow(ur[1],2)+pow(ur[2],2))/ur[0]);
	//calculate speeds of sound
	const a_real ci = sqrt(g*pi/ul[0]);
	const a_real cj = sqrt(g*pj/ur[0]);
	//calculate normal velocities
	const a_real vni = (ul[1]*n[0] + ul[2]*n[1])/ul[0];
	const a_real vnj = (ur[1]*n[0] + ur[2]*n[1])/ur[0];
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
	physics->evaluate_normal_jacobian(ul, n, dfdl);
	physics->evaluate_normal_jacobian(ur, n, dfdr);

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

void VanLeerFlux::get_flux(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur,
		const a_real* const __restrict__ n, a_real *const __restrict__ flux)
{
	a_real fiplus[NVARS], fjminus[NVARS];

	const a_real nx = n[0];
	const a_real ny = n[1];

	//calculate presures from u
	const a_real pi = (g-1)*(ul[3] - 0.5*(pow(ul[1],2)+pow(ul[2],2))/ul[0]);
	const a_real pj = (g-1)*(ur[3] - 0.5*(pow(ur[1],2)+pow(ur[2],2))/ur[0]);
	//calculate speeds of sound
	const a_real ci = sqrt(g*pi/ul[0]);
	const a_real cj = sqrt(g*pj/ur[0]);
	//calculate normal velocities
	const a_real vni = (ul[1]*nx +ul[2]*ny)/ul[0];
	const a_real vnj = (ur[1]*nx + ur[2]*ny)/ur[0];

	//Normal mach numbers
	const a_real Mni = vni/ci;
	const a_real Mnj = vnj/cj;

	//Calculate split fluxes
	if(Mni < -1.0)
		for(int i = 0; i < NVARS; i++)
			fiplus[i] = 0;
	else if(Mni > 1.0)
	{
		fiplus[0] = ul[0]*vni;
		fiplus[1] = vni*ul[1] + pi*nx;
		fiplus[2] = vni*ul[2] + pi*ny;
		fiplus[3] = vni*(ul[3] + pi);
	}
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
	{
		fjminus[0] = ur[0]*vnj;
		fjminus[1] = vnj*ur[1] + pj*nx;
		fjminus[2] = vnj*ur[2] + pj*ny;
		fjminus[3] = vnj*(ur[3] + pj);
	}
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

		for(int k = 0; k < NVARS; k++)
			dpL[k] = dML[k]*pi*(2.0-Mni) + ML*dpi[k]*(2.0-Mni) - ML*pi*dmni[k];
	}
	else if(Mni < -1.0) {
		ML = 0;
	}
	else {
		ML = Mni;
		for(int k = 0; k < NVARS; k++) {
			dML[k] = dmni[k];
			dpL[k] = dpi[k];
		}
	}
	
	if(fabs(Mnj) <= 1.0) {
		MR = -0.25*(Mnj-1)*(Mnj-1);
		for(int k = 0; k < NVARS; k++)
			dMR[k] = -0.5*(Mnj-1)*dmnj[k];

		for(int k = 0; k < NVARS; k++)
			dpR[k] = -dMR[k]*pj*(2.0+Mnj) - MR*dpj[k]*(2.0+Mnj) - MR*pj*dmnj[k];
	}
	else if(Mnj < -1.0) {
		MR = Mnj;
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
	// TODO
}

RoeFlux::RoeFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{ }

void RoeFlux::get_flux(const a_real *const ul, const a_real *const ur,
		const a_real* const n, a_real *const __restrict flux)
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
	// enthalpies  ( NOT E + p/rho = u(3)/u(0) + p/u(0) )
	//const a_real Hi = g/(g-1.0)* pi/ul[0] + 0.5*vmag2i;
	//const a_real Hj = g/(g-1.0)* pj/ur[0] + 0.5*vmag2j;
	const a_real Hi = (ul[3]+pi)/ul[0];
	const a_real Hj = (ur[3]+pj)/ur[0];

	//> compute Roe-averages
	
	const a_real Rij = sqrt(ur[0]/ul[0]);
	const a_real rhoij = Rij*ul[0];
	const a_real vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	const a_real vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	const a_real Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	const a_real vm2ij = vxij*vxij + vyij*vyij;
	const a_real vnij = vxij*n[0] + vyij*n[1];
	const a_real cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	//> eigenvalues
	
	a_real l[4];
	l[0] = fabs(vnij-cij); l[1] = fabs(vnij); l[2] = l[1]; l[3] = fabs(vnij+cij);
	
	// Harten entropy fix
	const a_real delta = 0.001*cij;
	// const a_real delta = 0.001*(vnj-vni);
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		if(l[ivar] < delta)
			l[ivar] = (l[ivar]*l[ivar] + delta*delta)/(2.0*delta);
	}

	//> A_Roe * dU
	
	const a_real dvn = vnj-vni, dp = pj-pi;
	a_real adu[NVARS];
	// product of eigenvalues and wave strengths (for component 2, sort of):
	a_real lalpha[NVARS];   
	lalpha[0] = l[0]*(dp-rhoij*cij*dvn)/(2.0*cij*cij);
	lalpha[1] = l[1]*(ur[0]-ul[0] - dp/(cij*cij));
	lalpha[2] = l[1]*rhoij;
	//lalpha[2] = l[1]*((vxj-vxi)*n[1]-(vyj-vyi)*n[0]);
	lalpha[3] = l[3]*(dp+rhoij*cij*dvn)/(2.0*cij*cij);

	// un-c:
	adu[0] = lalpha[0];
	adu[1] = lalpha[0]*(vxij-cij*n[0]);
	adu[2] = lalpha[0]*(vyij-cij*n[1]);
	adu[3] = lalpha[0]*(Hij-cij*vnij);

	// un:
	adu[0] += lalpha[1];
	adu[1] += lalpha[1]*vxij +      lalpha[2]*(vxj-vxi - dvn*n[0])/*cij*n[1]*/; 
	adu[2] += lalpha[1]*vyij +      lalpha[2]*(vyj-vyi - dvn*n[1])/*(-cij)*n[0]*/;
	adu[3] += lalpha[1]*vm2ij/2.0 + lalpha[2] *(vxij*(vxj-vxi) +vyij*(vyj-vyi) -vnij*dvn)
		/*cij*(vxij*n[1]-vyij*n[0])*/;

	// un+c:
	adu[0] += lalpha[3];
	adu[1] += lalpha[3]*(vxij+cij*n[0]);
	adu[2] += lalpha[3]*(vyij+cij*n[1]);
	adu[3] += lalpha[3]*(Hij+cij*vnij);

	// get one-sided flux vectors
	a_real fi[4], fj[4];
	fi[0] = ul[0]*vni;						fj[0] = ur[0]*vnj;
	fi[1] = ul[0]*vni*vxi + pi*n[0];		fj[1] = ur[0]*vnj*vxj + pj*n[0];
	fj[2] = ul[0]*vni*vyi + pi*n[1];		fj[2] = ur[0]*vnj*vyj + pj*n[1];
	fj[3] = vni*(ul[3] + pi);				fj[3] = vnj*(ur[3] + pj);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - adu[ivar]);
	}
}

void RoeFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	std::cout << " ! RoeFlux: Not implemented!\n";
}

HLLFlux::HLLFlux(const IdealGasPhysics *const analyticalflux) 
	: InviscidFlux(analyticalflux)
{
}

/** \cite invflux_batten
 */
void HLLFlux::get_flux(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, 
		const a_real* const __restrict__ n, a_real *const __restrict__ flux)
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
	physics->evaluate_normal_jacobian(ul, n, dfdl);
	physics->evaluate_normal_jacobian(ur, n, dfdr);
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
	: InviscidFlux(analyticalflux)
{
}

/** Currently, the estimated signal speeds are the Einfeldt estimates, 
 * not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
		a_real *const __restrict flux)
{
	a_real utemp[NVARS];

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
	//rhoij = Rij*ul[0];
	const a_real vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	const a_real vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	const a_real Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	const a_real vm2ij = vxij*vxij + vyij*vyij;
	const a_real vnij = vxij*n[0] + vyij*n[1];
	const a_real cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	// estimate signal speeds (classical; not Remaki corrected)
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
	{
		flux[0] = vni*ul[0];
		flux[1] = vni*ul[1] + pi*n[0];
		flux[2] = vni*ul[2] + pi*n[1];
		flux[3] = vni*(ul[3] + pi);
	}
	else if(sl <= 0 && sm > 0)
	{
		flux[0] = vni*ul[0];
		flux[1] = vni*ul[1] + pi*n[0];
		flux[2] = vni*ul[2] + pi*n[1];
		flux[3] = vni*(ul[3] + pi);

		const a_real pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
		utemp[0] = ul[0] * (sl - vni)/(sl-sm);
		utemp[1] = ( (sl-vni)*ul[1] + (pstar-pi)*n[0] )/(sl-sm);
		utemp[2] = ( (sl-vni)*ul[2] + (pstar-pi)*n[1] )/(sl-sm);
		utemp[3] = ( (sl-vni)*ul[3] - pi*vni + pstar*sm )/(sl-sm);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sl * ( utemp[ivar] - ul[ivar]);
	}
	else if(sm <= 0 && sr >= 0)
	{
		flux[0] = vnj*ur[0];
		flux[1] = vnj*ur[1] + pj*n[0];
		flux[2] = vnj*ur[2] + pj*n[1];
		flux[3] = vnj*(ur[3] + pj);

		const a_real pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
		utemp[0] = ur[0] * (sr - vnj)/(sr-sm);
		utemp[1] = ( (sr-vnj)*ur[1] + (pstar-pj)*n[0] )/(sr-sm);
		utemp[2] = ( (sr-vnj)*ur[2] + (pstar-pj)*n[1] )/(sr-sm);
		utemp[3] = ( (sr-vnj)*ur[3] - pj*vnj + pstar*sm )/(sr-sm);

		for(int ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sr * ( utemp[ivar] - ur[ivar]);
	}
	else
	{
		flux[0] = vnj*ur[0];
		flux[1] = vnj*ur[1] + pj*n[0];
		flux[2] = vnj*ur[2] + pj*n[1];
		flux[3] = vnj*(ur[3] + pj);
	}
}

/*
  Differentiation of HLLC_flux in forward (tangent) mode:
   variations   of useful results: *flux
   with respect to varying inputs: *ul
   RW status of diff variables: *flux:out *ul:in
   Plus diff mem management of: flux:in ul:in
   Multidirectional mode

 * The estimated signal speeds are the Einfeldt estimates, 
 * not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::getFluxJac_left(const a_real *const ul, const a_real *const ur, 
		const a_real *const n, 
		a_real *const __restrict flux, a_real *const __restrict fluxd) 
{
    a_real uld[NVARS][NVARS];
	for(int i = 0; i < NVARS; i++) {
		for(int j = 0; j < NVARS; j++)
			uld[i][j] = 0;
		uld[i][i] = 1.0;
	}
	
    a_real utemp[NVARS];
    a_real utempd[NVARS][NVARS];
    a_real vxid[NVARS];
    int ii1;
    const a_real vxi = ul[1]/ul[0];
    a_real vyid[NVARS];
    const a_real vyi = ul[2]/ul[0];
    a_real vmag2id[NVARS];
    const a_real vmag2i = vxi*vxi + vyi*vyi;
    a_real pid[NVARS];
    const a_real pi = (g-1.0)*(ul[3]-0.5*ul[0]*vmag2i);
    a_real arg1d[NVARS];
    int nd;
    a_real arg1;
    arg1 = g*pi/ul[0];
    a_real vnid[NVARS];
    a_real cid[NVARS];
    a_real Hid[NVARS];
    a_real srd[NVARS], sld[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        vxid[nd] = (uld[1][nd]*ul[0]-ul[1]*uld[0][nd])/(ul[0]*ul[0]);
        vyid[nd] = (uld[2][nd]*ul[0]-ul[2]*uld[0][nd])/(ul[0]*ul[0]);
        vnid[nd] = n[0]*vxid[nd] + n[1]*vyid[nd];
        vmag2id[nd] = vxid[nd]*vxi + vxi*vxid[nd] + vyid[nd]*vyi + vyi*vyid[nd];
        pid[nd] = (g-1.0)*(uld[3][nd]-0.5*(uld[0][nd]*vmag2i+ul[0]*vmag2id[nd]));
        arg1d[nd] = (g*pid[nd]*ul[0]-g*pi*uld[0][nd])/(ul[0]*ul[0]);
        cid[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        Hid[nd] = ((uld[3][nd]+pid[nd])*ul[0]-(ul[3]+pi)*uld[0][nd])/(ul[0]*ul[0]);
        arg1d[nd] = -(ur[0]*uld[0][nd]/(ul[0]*ul[0]));
        sld[nd] = vnid[nd] - cid[nd];
    }
    const a_real vxj = ur[1]/ur[0];
    const a_real vyj = ur[2]/ur[0];
    const a_real vni = vxi*n[0] + vyi*n[1];
    const a_real vnj = vxj*n[0] + vyj*n[1];
    const a_real vmag2j = vxj*vxj + vyj*vyj;
    const a_real pj = (g-1.0)*(ur[3]-0.5*ur[0]*vmag2j);
    const a_real ci = sqrt(arg1);
    arg1 = g*pj/ur[0];
    const a_real cj = sqrt(arg1);
    const a_real Hi = (ul[3]+pi)/ul[0];
    const a_real Hj = (ur[3]+pj)/ur[0];
    arg1 = ur[0]/ul[0];
    a_real Rijd[NVARS];
    const a_real Rij = sqrt(arg1);
    a_real vxijd[NVARS];
    const a_real vxij = (Rij*vxj+vxi)/(Rij+1.0);
    a_real vyijd[NVARS];
    a_real vm2ijd[NVARS];
    a_real vnijd[NVARS];
    a_real Hijd[NVARS];
    const a_real vyij = (Rij*vyj+vyi)/(Rij+1.0);
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        vxijd[nd] = ((vxj*Rijd[nd]+vxid[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((vyj*Rijd[nd]+vyid[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        Hijd[nd] = ((Hj*Rijd[nd]+Hid[nd])*(Rij+1.0)-(Rij*Hj+Hi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vm2ijd[nd] = vxijd[nd]*vxij + vxij*vxijd[nd] + vyijd[nd]*vyij + vyij*vyijd[nd];
        vnijd[nd] = n[0]*vxijd[nd] + n[1]*vyijd[nd];
        arg1d[nd] = (g-1.0)*(Hijd[nd]-0.5*vm2ijd[nd]);
    }
    const a_real Hij = (Rij*Hj+Hi)/(Rij+1.0);
    const a_real vm2ij = vxij*vxij + vyij*vyij;
    const a_real vnij = vxij*n[0] + vyij*n[1];
    arg1 = (g-1.0)*(Hij-vm2ij*0.5);
    a_real cijd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        cijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
    const a_real cij = sqrt(arg1);
    
	// estimate signal speeds (classical; not Remaki corrected)
    a_real sr, sl;
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
    a_real sm;
    a_real smd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        smd[nd] = ((ur[0]*vnj*srd[nd]-(uld[0][nd]*vni+ul[0]*vnid[nd])*(sl-vni)
            -ul[0]*vni*(sld[nd]-vnid[nd])+pid[nd])*(ur[0]*(sr-vnj)-ul[0]*(sl-
            vni))-(ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)*(ur[0]*srd[nd]
            -uld[0][nd]*(sl-vni)-ul[0]*(sld[nd]-vnid[nd])))/((ur[0]*(sr-vnj)-
            ul[0]*(sl-vni))*(ur[0]*(sr-vnj)-ul[0]*(sl-vni)));
    sm = (ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)/(ur[0]*(sr-vnj)-ul[0]*(
        sl-vni));
    // compute fluxes
    if (sl > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnid[nd]*ul[0] + vni*uld[0][nd];
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnid[nd]*ul[1] + vni*uld[1][nd] + n[0]*pid[nd];
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnid[nd]*ul[2] + vni*uld[2][nd] + n[1]*pid[nd];
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnid[nd]*(ul[3]+pi) + vni*(uld[3][nd]+pid[nd]);
        flux[3] = vni*(ul[3]+pi);
    } else if (sl <= 0 && sm > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnid[nd]*ul[0] + vni*uld[0][nd];
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnid[nd]*ul[1] + vni*uld[1][nd] + n[0]*pid[nd];
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnid[nd]*ul[2] + vni*uld[2][nd] + n[1]*pid[nd];
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnid[nd]*(ul[3]+pi) + vni*(uld[3][nd]+pid[nd]);
        flux[3] = vni*(ul[3]+pi);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = (uld[0][nd]*(vni-sl)+ul[0]*(vnid[nd]-sld[nd]))*(vni-
                sm) + ul[0]*(vni-sl)*(vnid[nd]-smd[nd]) + pid[nd];
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
            utempd[0][nd] = ((uld[0][nd]*(sl-vni)+ul[0]*(sld[nd]-vnid[nd]))*(
                sl-sm)-ul[0]*(sl-vni)*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        }
        pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
        utemp[0] = ul[0]*(sl-vni)/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = (((sld[nd]-vnid[nd])*ul[1]+(sl-vni)*uld[1][nd]+n[0
                ]*(pstard[nd]-pid[nd]))*(sl-sm)-((sl-vni)*ul[1]+(pstar-pi)*n[0
                ])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[1] = ((sl-vni)*ul[1]+(pstar-pi)*n[0])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = (((sld[nd]-vnid[nd])*ul[2]+(sl-vni)*uld[2][nd]+n[1
                ]*(pstard[nd]-pid[nd]))*(sl-sm)-((sl-vni)*ul[2]+(pstar-pi)*n[1
                ])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[2] = ((sl-vni)*ul[2]+(pstar-pi)*n[1])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = (((sld[nd]-vnid[nd])*ul[3]+(sl-vni)*uld[3][nd]-pid
                [nd]*vni-pi*vnid[nd]+pstard[nd]*sm+pstar*smd[nd])*(sl-sm)-((sl
                -vni)*ul[3]-pi*vni+pstar*sm)*(sld[nd]-smd[nd]))/((sl-sm)*(sl-
                sm));
        utemp[3] = ((sl-vni)*ul[3]-pi*vni+pstar*sm)/(sl-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + sld[nd]*(utemp[ivar]-ul[
                    ivar]) + sl*(utempd[ivar][nd]-uld[ivar][nd]);
            flux[ivar] += sl*(utemp[ivar]-ul[ivar]);
        }
    } else if (sm <= 0 && sr >= 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd) {
            fluxd[3*NVARS+nd] = 0.0;
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
        }
        flux[3] = vnj*(ur[3]+pj);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = ur[0]*(-(srd[nd]*(vnj-sm))-(vnj-sr)*smd[nd]);
            utempd[0][nd] = (ur[0]*srd[nd]*(sr-sm)-ur[0]*(sr-vnj)*(srd[nd]-smd
                [nd]))/((sr-sm)*(sr-sm));
        }
        pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
        utemp[0] = ur[0]*(sr-vnj)/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = ((ur[1]*srd[nd]+n[0]*pstard[nd])*(sr-sm)-((sr-vnj)
                *ur[1]+(pstar-pj)*n[0])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[1] = ((sr-vnj)*ur[1]+(pstar-pj)*n[0])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = ((ur[2]*srd[nd]+n[1]*pstard[nd])*(sr-sm)-((sr-vnj)
                *ur[2]+(pstar-pj)*n[1])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[2] = ((sr-vnj)*ur[2]+(pstar-pj)*n[1])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = ((ur[3]*srd[nd]+pstard[nd]*sm+pstar*smd[nd])*(sr-
                sm)-((sr-vnj)*ur[3]-pj*vnj+pstar*sm)*(srd[nd]-smd[nd]))/((sr-
                sm)*(sr-sm));
        utemp[3] = ((sr-vnj)*ur[3]-pj*vnj+pstar*sm)/(sr-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + srd[nd]*(utemp[ivar]-ur[
                    ivar]) + sr*utempd[ivar][nd];
            flux[ivar] += sr*(utemp[ivar]-ur[ivar]);
        }
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = 0.0;
        flux[3] = vnj*(ur[3]+pj);
    }
}

/*
  Differentiation of get_HLLC_flux in forward (tangent) mode:
   variations   of useful results: *flux
   with respect to varying inputs: *ur
   RW status of diff variables: *flux:out *ur:in
   Plus diff mem management of: flux:in ur:in
   Multidirectional mode

 * Currently, the estimated signal speeds are the Einfeldt estimates, 
 * not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::getFluxJac_right(const a_real *const ul, const a_real *const ur, 
		const a_real *const n, 
		a_real *const __restrict flux, a_real *const __restrict fluxd) 
{
    a_real urd[NVARS][NVARS];
	for(int i = 0; i < NVARS; i++) {
		for(int j = 0; j < NVARS; j++)
			urd[i][j] = 0;
		urd[i][i] = 1.0;
	}
	
    a_real utemp[NVARS];
    a_real utempd[NVARS][NVARS];
    int ii1;
    const a_real vxi = ul[1]/ul[0];
    const a_real vyi = ul[2]/ul[0];
    a_real vxjd[NVARS];
    const a_real vxj = ur[1]/ur[0];
    a_real vyjd[NVARS];
    const a_real vyj = ur[2]/ur[0];
    const a_real vmag2i = vxi*vxi + vyi*vyi;
    a_real vmag2jd[NVARS];
    const a_real vmag2j = vxj*vxj + vyj*vyj;
    const a_real pi = (g-1.0)*(ul[3]-0.5*ul[0]*vmag2i);
    a_real pjd[NVARS];
    const a_real pj = (g-1.0)*(ur[3]-0.5*ur[0]*vmag2j);
    a_real arg1d[NVARS];
    int nd;
    a_real arg1;
    arg1 = g*pi/ul[0];
    const a_real ci = sqrt(arg1);
    arg1 = g*pj/ur[0];
    a_real vnjd[NVARS];
    a_real cjd[NVARS];
    a_real Hjd[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        vxjd[nd] = (urd[1][nd]*ur[0]-ur[1]*urd[0][nd])/(ur[0]*ur[0]);
        vyjd[nd] = (urd[2][nd]*ur[0]-ur[2]*urd[0][nd])/(ur[0]*ur[0]);
        vnjd[nd] = n[0]*vxjd[nd] + n[1]*vyjd[nd];
        vmag2jd[nd] = vxjd[nd]*vxj + vxj*vxjd[nd] + vyjd[nd]*vyj + vyj*vyjd[nd];
        pjd[nd] = (g-1.0)*(urd[3][nd]-0.5*(urd[0][nd]*vmag2j+ur[0]*vmag2jd[nd]));
        arg1d[nd] = (g*pjd[nd]*ur[0]-g*pj*urd[0][nd])/(ur[0]*ur[0]);
        cjd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        Hjd[nd] = ((urd[3][nd]+pjd[nd])*ur[0]-(ur[3]+pj)*urd[0][nd])/(ur[0]*ur[0]);
        arg1d[nd] = urd[0][nd]/ul[0];
    }
    const a_real vni = vxi*n[0] + vyi*n[1];
    const a_real vnj = vxj*n[0] + vyj*n[1];
    const a_real cj = sqrt(arg1);
    const a_real Hi = (ul[3]+pi)/ul[0];
    const a_real Hj = (ur[3]+pj)/ur[0];
    arg1 = ur[0]/ul[0];
    a_real Rijd[NVARS];
    const a_real Rij = sqrt(arg1);
    a_real vxijd[NVARS];
    const a_real vxij = (Rij*vxj+vxi)/(Rij+1.0);
    a_real vyijd[NVARS];
    const a_real vyij = (Rij*vyj+vyi)/(Rij+1.0);
    a_real Hijd[NVARS];
    a_real vm2ijd[NVARS];
    a_real vnijd[NVARS];
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
        vxijd[nd] = ((Rijd[nd]*vxj+Rij*vxjd[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((Rijd[nd]*vyj+Rij*vyjd[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])
			/((Rij+1.0)*(Rij+1.0));
        Hijd[nd] = ((Rijd[nd]*Hj+Rij*Hjd[nd])*(Rij+1.0)-(Rij*Hj+Hi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vm2ijd[nd] = vxijd[nd]*vxij + vxij*vxijd[nd] + vyijd[nd]*vyij + vyij*vyijd[nd];
        vnijd[nd] = n[0]*vxijd[nd] + n[1]*vyijd[nd];
        arg1d[nd] = (g-1.0)*(Hijd[nd]-0.5*vm2ijd[nd]);
    }
    const a_real Hij = (Rij*Hj+Hi)/(Rij+1.0);
    const a_real vm2ij = vxij*vxij + vyij*vyij;
    const a_real vnij = vxij*n[0] + vyij*n[1];
    arg1 = (g-1.0)*(Hij-vm2ij*0.5);
    a_real cijd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        cijd[nd] = fabs(arg1)<ZERO_TOL ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1));
    const a_real cij = sqrt(arg1);
    // estimate signal speeds (classical; not Remaki corrected)
    a_real sr, sl;
    a_real srd[NVARS], sld[NVARS];
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
    a_real sm;
    a_real smd[NVARS];
    for (nd = 0; nd < NVARS; ++nd)
        smd[nd] = (((urd[0][nd]*vnj+ur[0]*vnjd[nd])*(sr-vnj)+ur[0]*vnj*(srd[nd
            ]-vnjd[nd])-ul[0]*vni*sld[nd]-pjd[nd])*(ur[0]*(sr-vnj)-ul[0]*(sl-
            vni))-(ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)*(urd[0][nd]*(
            sr-vnj)+ur[0]*(srd[nd]-vnjd[nd])-ul[0]*sld[nd]))/((ur[0]*(sr-vnj)-
            ul[0]*(sl-vni))*(ur[0]*(sr-vnj)-ul[0]*(sl-vni)));
    sm = (ur[0]*vnj*(sr-vnj)-ul[0]*vni*(sl-vni)+pi-pj)/(ur[0]*(sr-vnj)-ul[0]*(
        sl-vni));
    // compute fluxes
    if (sl > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = 0.0;
        flux[3] = vni*(ul[3]+pi);
    } else if (sl <= 0 && sm > 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = 0.0;
        flux[0] = vni*ul[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = 0.0;
        flux[1] = vni*ul[1] + pi*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = 0.0;
        flux[2] = vni*ul[2] + pi*n[1];
        for (nd = 0; nd < NVARS; ++nd) {
            fluxd[3*NVARS+nd] = 0.0;
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
        }
        flux[3] = vni*(ul[3]+pi);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = ul[0]*(-(sld[nd]*(vni-sm))-(vni-sl)*smd[nd]);
            utempd[0][nd] = (ul[0]*sld[nd]*(sl-sm)-ul[0]*(sl-vni)*(sld[nd]-smd
                [nd]))/((sl-sm)*(sl-sm));
        }
        pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
        utemp[0] = ul[0]*(sl-vni)/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = ((ul[1]*sld[nd]+n[0]*pstard[nd])*(sl-sm)-((sl-vni)
                *ul[1]+(pstar-pi)*n[0])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[1] = ((sl-vni)*ul[1]+(pstar-pi)*n[0])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = ((ul[2]*sld[nd]+n[1]*pstard[nd])*(sl-sm)-((sl-vni)
                *ul[2]+(pstar-pi)*n[1])*(sld[nd]-smd[nd]))/((sl-sm)*(sl-sm));
        utemp[2] = ((sl-vni)*ul[2]+(pstar-pi)*n[1])/(sl-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = ((ul[3]*sld[nd]+pstard[nd]*sm+pstar*smd[nd])*(sl-
                sm)-((sl-vni)*ul[3]-pi*vni+pstar*sm)*(sld[nd]-smd[nd]))/((sl-
                sm)*(sl-sm));
        utemp[3] = ((sl-vni)*ul[3]-pi*vni+pstar*sm)/(sl-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + sld[nd]*(utemp[ivar]-ul[
                    ivar]) + sl*utempd[ivar][nd];
            flux[ivar] += sl*(utemp[ivar]-ul[ivar]);
        }
    } else if (sm <= 0 && sr >= 0) {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnjd[nd]*ur[0] + vnj*urd[0][nd];
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnjd[nd]*ur[1] + vnj*urd[1][nd] + n[0]*pjd[nd];
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnjd[nd]*ur[2] + vnj*urd[2][nd] + n[1]*pjd[nd];
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnjd[nd]*(ur[3]+pj) + vnj*(urd[3][nd]+pjd[nd]);
        flux[3] = vnj*(ur[3]+pj);
        a_real pstar;
        a_real pstard[NVARS];
        for (nd = 0; nd < NVARS; ++nd) {
            pstard[nd] = (urd[0][nd]*(vnj-sr)+ur[0]*(vnjd[nd]-srd[nd]))*(vnj-
                sm) + ur[0]*(vnj-sr)*(vnjd[nd]-smd[nd]) + pjd[nd];
            for (ii1 = 0; ii1 < NVARS; ++ii1)
                utempd[ii1][nd] = 0.0;
            utempd[0][nd] = ((urd[0][nd]*(sr-vnj)+ur[0]*(srd[nd]-vnjd[nd]))*(
                sr-sm)-ur[0]*(sr-vnj)*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        }
        pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
        utemp[0] = ur[0]*(sr-vnj)/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[1][nd] = (((srd[nd]-vnjd[nd])*ur[1]+(sr-vnj)*urd[1][nd]+n[0
                ]*(pstard[nd]-pjd[nd]))*(sr-sm)-((sr-vnj)*ur[1]+(pstar-pj)*n[0
                ])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[1] = ((sr-vnj)*ur[1]+(pstar-pj)*n[0])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[2][nd] = (((srd[nd]-vnjd[nd])*ur[2]+(sr-vnj)*urd[2][nd]+n[1
                ]*(pstard[nd]-pjd[nd]))*(sr-sm)-((sr-vnj)*ur[2]+(pstar-pj)*n[1
                ])*(srd[nd]-smd[nd]))/((sr-sm)*(sr-sm));
        utemp[2] = ((sr-vnj)*ur[2]+(pstar-pj)*n[1])/(sr-sm);
        for (nd = 0; nd < NVARS; ++nd)
            utempd[3][nd] = (((srd[nd]-vnjd[nd])*ur[3]+(sr-vnj)*urd[3][nd]-pjd
                [nd]*vnj-pj*vnjd[nd]+pstard[nd]*sm+pstar*smd[nd])*(sr-sm)-((sr
                -vnj)*ur[3]-pj*vnj+pstar*sm)*(srd[nd]-smd[nd]))/((sr-sm)*(sr-
                sm));
        utemp[3] = ((sr-vnj)*ur[3]-pj*vnj+pstar*sm)/(sr-sm);
        for (int ivar = 0; ivar < NVARS; ++ivar) {
            for (nd = 0; nd < NVARS; ++nd)
                fluxd[ivar*NVARS+nd] = fluxd[ivar*NVARS+nd] + srd[nd]*(utemp[ivar]-ur[
                    ivar]) + sr*(utempd[ivar][nd]-urd[ivar][nd]);
            flux[ivar] += sr*(utemp[ivar]-ur[ivar]);
        }
    } else {
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[0*NVARS+nd] = vnjd[nd]*ur[0] + vnj*urd[0][nd];
        flux[0] = vnj*ur[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[1*NVARS+nd] = vnjd[nd]*ur[1] + vnj*urd[1][nd] + n[0]*pjd[nd];
        flux[1] = vnj*ur[1] + pj*n[0];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[2*NVARS+nd] = vnjd[nd]*ur[2] + vnj*urd[2][nd] + n[1]*pjd[nd];
        flux[2] = vnj*ur[2] + pj*n[1];
        for (nd = 0; nd < NVARS; ++nd)
            fluxd[3*NVARS+nd] = vnjd[nd]*(ur[3]+pj) + vnj*(urd[3][nd]+pjd[nd]);
        flux[3] = vnj*(ur[3]+pj);
    }
}

void HLLCFlux::get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
		a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	std::cout << " !!!! HLLC Jacobian not available!!\n";
	a_real flux[NVARS];
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

void HLLCFlux::get_flux_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, 
		a_real *const __restrict flux, a_real *const __restrict dfdl, a_real *const __restrict dfdr)
{
	std::cout << " !!!! HLLC Jacobian not available!!\n";
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

} // end namespace acfd

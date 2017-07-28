/** \file anumericalflux.cpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#include "anumericalflux.hpp"

namespace acfd {

InviscidFlux::InviscidFlux(const a_real gamma, const EulerFlux *const analyticalflux) 
	: g(gamma), aflux(analyticalflux)
{ }

void InviscidFlux::get_jacobian(const a_real *const uleft, const a_real *const uright, 
		const a_real* const n, 
		a_real *const dfdl, a_real *const dfdr)
{ }

InviscidFlux::~InviscidFlux()
{ }

LocalLaxFriedrichsFlux::LocalLaxFriedrichsFlux(const a_real gamma, const EulerFlux *const analyticalflux)
	: InviscidFlux(gamma, analyticalflux)
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
	const a_real eig = std::fabs(vni)+ci > std::fabs(vnj)+cj ? std::fabs(vni)+ci : std::fabs(vnj)+cj;
	
	flux[0] = 0.5*( ul[0]*vni + ur[0]*vnj - eig*(ur[0]-ul[0]) );
	flux[1] = 0.5*( vni*ul[1]+pi*n[0] + vnj*ur[1]+pj*n[0] - eig*(ur[1]-ul[1]) );
	flux[2] = 0.5*( vni*ul[2]+pi*n[1] + vnj*ur[2]+pj*n[1] - eig*(ur[2]-ul[2]) );
	flux[3] = 0.5*( vni*(ul[3]+pi) + vnj*(ur[3]+pj) - eig*(ur[3] - ul[3]) );
}

void LocalLaxFriedrichsFlux::get_jacobian(const a_real *const __restrict__ ul, 
		const a_real *const __restrict__ ur,
		const a_real* const __restrict__ n, 
		a_real *const __restrict__ dfdl, a_real *const __restrict__ dfdr)
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
	//bool leftismax;
	if( std::fabs(vni)+ci >= std::fabs(vnj)+cj )
	{
		eig = std::fabs(vni)+ci;
		//leftismax = true;
	}
	else
	{
		eig = std::fabs(vnj)+cj;
		//leftismax = false;
	}

	// get flux jacobians
	aflux->evaluate_normal_jacobian(ul, n, dfdl);
	aflux->evaluate_normal_jacobian(ur, n, dfdr);

	// linearization of the dissipation term
	
	/*const a_real ctermi = 0.5 / std::sqrt( g*(g-1)/ul[0]* (ul[3]-(ul[1]*ul[1]+ul[2]*ul[2])/(2*ul[0])) );
	const a_real ctermj = 0.5 / std::sqrt( g*(g-1)/ur[0]* (ur[3]-(ur[1]*ur[1]+ur[2]*ur[2])/(2*ur[0])) );
	a_real dedu[NVARS];
	
	if(leftismax) {
		dedu[0] = -std::fabs(vni/(ul[0]*ul[0])) + ctermi*g*(g-1)*( -ul[3]/(ul[0]*ul[0]) + (ul[1]*ul[1]+ul[2]*ul[2])/(ul[0]*ul[0]*ul[0]) );
		dedu[1] = (vni>0 ? n[0]/ul[0] : -n[0]/ul[0]) + ctermi*g*(g-1)*(-ul[1]/ul[0]);
		dedu[2] = (vni>0 ? n[1]/ul[0] : -n[1]/ul[0]) + ctermi*g*(g-1)*(-ul[2]/ul[0]);
		dedu[3] = ctermi*g*(g-1)/ul[0];
	} 
	else {
		for(int i = 0; i < NVARS; i++)
			dedu[i] = 0;
	}*/

	// add contributions to left derivative
	for(int i = 0; i < NVARS; i++)
	{
		dfdl[i*NVARS+i] -= -eig;
		/*for(int j = 0; j < NVARS; j++)
			dfdl[i*NVARS+j] -= dedu[j]*(ur[i]-ul[i]);*/
	}

	/*if(leftismax) {
		for(int i = 0; i < NVARS; i++)
			dedu[i] = 0;
	} else {
		dedu[0] = -std::fabs(vnj/(ur[0]*ur[0])) + ctermj*g*(g-1)*( -ur[3]/(ur[0]*ur[0]) + (ur[1]*ur[1]+ur[2]*ur[2])/(ur[0]*ur[0]*ur[0]) );
		dedu[1] = (vnj>0 ? n[0]/ur[0] : -n[0]/ur[0]) + ctermj*g*(g-1)*(-ur[1]/ur[0]);
		dedu[2] = (vnj>0 ? n[1]/ur[0] : -n[1]/ur[0]) + ctermj*g*(g-1)*(-ur[2]/ur[0]);
		dedu[3] = ctermj*g*(g-1)/ur[0];
	}*/

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

void LLF_get_jacobian(const a_real *const __restrict__ ul, 
		const a_real *const __restrict__ ur,
		const a_real* const __restrict__ n, 
		a_real *const __restrict__ dfdl, a_real *const __restrict__ dfdr)
{
	a_real eig; 
	a_real g=-1000000000000000000;

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
	//aflux->evaluate_normal_jacobian(ul, n, dfdl);
	//aflux->evaluate_normal_jacobian(ur, n, dfdr);

	// linearization of the dissipation term
	
	const a_real ctermi = 0.5 / std::sqrt( g*(g-1)/ul[0]* (ul[3]-(ul[1]*ul[1]+ul[2]*ul[2])/(2*ul[0])) );
	const a_real ctermj = 0.5 / std::sqrt( g*(g-1)/ur[0]* (ur[3]-(ur[1]*ur[1]+ur[2]*ur[2])/(2*ur[0])) );
	a_real dedu[NVARS];
	
	if(leftismax) {
		dedu[0] = -std::fabs(vni/(ul[0]*ul[0])) + ctermi*g*(g-1)*( -ul[3]/(ul[0]*ul[0]) + (ul[1]*ul[1]+ul[2]*ul[2])/(ul[0]*ul[0]*ul[0]) );
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
		dedu[0] = -std::fabs(vnj/(ur[0]*ur[0])) + ctermj*g*(g-1)*( -ur[3]/(ur[0]*ur[0]) + (ur[1]*ur[1]+ur[2]*ur[2])/(ur[0]*ur[0]*ur[0]) );
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

VanLeerFlux::VanLeerFlux(const a_real gamma, const EulerFlux *const analyticalflux) 
	: InviscidFlux(gamma, analyticalflux)
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
		const a_real vmags = pow(ur[1]/ur[0], 2) + pow(ur[2]/ur[0], 2);	// square of velocity magnitude
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

RoeFlux::RoeFlux(const a_real gamma, const EulerFlux *const analyticalflux) 
	: InviscidFlux(gamma, analyticalflux)
{ }

void RoeFlux::get_flux(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur,
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
	// enthalpies  ( NOT E + p/rho = u(3)/u(0) + p/u(0) )
	const a_real Hi = g/(g-1.0)* pi/ul[0] + 0.5*vmag2i;
	const a_real Hj = g/(g-1.0)* pj/ur[0] + 0.5*vmag2j;

	// compute Roe-averages
	
	const a_real Rij = sqrt(ur[0]/ul[0]);
	const a_real rhoij = Rij*ul[0];
	const a_real vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	const a_real vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	const a_real Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	const a_real vm2ij = vxij*vxij + vyij*vyij;
	const a_real vnij = vxij*n[0] + vyij*n[1];
	const a_real cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	// eigenvalues
	a_real l[4];
	l[0] = vnij; l[1] = vnij; l[2] = vnij + cij; l[3] = vnij - cij;

	// Harten-Hyman entropy fix
	a_real eps = 0;
	if(eps < l[0]-vni) eps = l[0]-vni;
	if(eps < vnj-l[0]) eps = vnj-l[0];
	if(fabs(l[0]) < eps) l[0] = eps;
	if(fabs(l[1]) < eps) l[1] = eps;

	eps = 0;
	if(eps < l[2]-(vni+ci)) eps = l[2]-(vni+ci);
	if(eps < vnj+cj - l[2]) eps = vnj+cj - l[2];
	if(fabs(l[2]) < eps) l[2] = eps;

	eps = 0;
	if(eps < l[3] - (vni-ci)) eps = l[3] - (vni-ci);
	if(eps < vnj-cj - l[3]) eps = vnj-cj - l[3];
	if(fabs(l[3]) < eps) l[3] = eps;

	// eigenvectors (column vectors of r below)
	a_real r[4][4];
	
	// according to Dr Luo's notes
	r[0][0] = 1.0;		r[0][1] = 0;							r[0][2] = 1.0;				r[0][3] = 1.0;
	r[1][0] = vxij;		r[1][1] = cij*n[1];						r[1][2] = vxij + cij*n[0];	r[1][3] = vxij - cij*n[0];
	r[2][0] = vyij;		r[2][1] = -cij*n[0];					r[2][2] = vyij + cij*n[1];	r[2][3] = vyij - cij*n[1];
	r[3][0]= vm2ij*0.5;	r[3][1] = cij*(vxij*n[1]-vyij*n[0]);	r[3][2] = Hij + cij*vnij;	r[3][3] = Hij - cij*vnij;

	// according to Fink (just a hack to make the overall flux equal the Roe flux in Fink's paper;
	// the second eigenvector is obviously not what is given below
	/*r(0,0) = 1.0;		r(0,2) = 1.0;				r(0,3) = 1.0;
	r(1,0) = vxij;		r(1,2) = vxij + cij*n[0];	r(1,3) = vxij - cij*n[0];
	r(2,0) = vyij;		r(2,2) = vyij + cij*n[1];	r(2,3) = vyij - cij*n[1];
	r(3,0)= vm2ij*0.5;	r(3,2) = Hij + cij*vnij;	r(3,3) = Hij - cij*vnij;
	
	r(0,1) = 0.0;
	r(1,1) = (vxj-vxi) - n[0]*(vnj-vni);
	r(2,1) = (vyj-vyi) - n[1]*(vnj-vni);
	r(3,1) = vxij*(vxj-vxi) + vyij*(vyj-vyi) - vnij*(vnj-vni);*/
	
	for(int ivar = 0; ivar < 4; ivar++)
	{
		r[ivar][2] *= rhoij/(2.0*cij);
		r[ivar][3] *= rhoij/(2.0*cij);
	}

	// R^(-1)(qR-qL)
	a_real dw[4];
	dw[0] = (ur[0]-ul[0]) - (pj-pi)/(cij*cij);
	dw[1] = (vxj-vxi)*n[1] - (vyj-vyi)*n[0];			// Dr Luo
	//dw(1) = rhoij;										// hack for conformance with Fink
	dw[2] = vnj-vni + (pj-pi)/(rhoij*cij);
	dw[3] = -(vnj-vni) + (pj-pi)/(rhoij*cij);

	// get one-sided flux vectors
	a_real fi[4], fj[4];
	fi[0] = ul[0]*vni;						fj[0] = ur[0]*vnj;
	fi[1] = ul[0]*vni*vxi + pi*n[0];		fj[1] = ur[0]*vnj*vxj + pj*n[0];
	fj[2] = ul[0]*vni*vyi + pi*n[1];		fj[2] = ur[0]*vnj*vyj + pj*n[1];
	fj[3] = vni*(ul[3] + pi);				fj[3] = vnj*(ur[3] + pj);

	// finally compute fluxes
	for(int ivar = 0; ivar < NVARS; ivar++)
	{
		a_real sum = 0;
		for(int j = 0; j < NVARS; j++)
			sum += fabs(l[j])*dw[j]*r[ivar][j];
		flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - sum);
	}
}

void RoeFlux::get_jacobian(const a_real *const ul, const a_real *const ur, 
		const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	std::cout << " ! RoeFlux: Not implemented!\n";
}

HLLFlux::HLLFlux(const a_real gamma, const EulerFlux *const analyticalflux) 
	: InviscidFlux(gamma, analyticalflux)
{
}

/** Ref: P. Batten, M.A. Lechziner, U.C. Goldberg. Average-state Jacobians and implicit methods 
 * for compressible viscous and turbulent flows.
 * JCP 137, pages 38--78. 1997.
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
	// enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
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

/** Automatically differentiated Jacobian w.r.t. left state, generated by Tapenade 3.12 (r6213) - 13 Oct 2016 10:54.
 * Modified to remove the runtime parameter nbdirs and the change in ul. Also changed the array shape of Jacobian.
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
    a_real Hid[NVARS], cid[NVARS], pid[NVARS], vxid[NVARS], vyid[NVARS], vmag2id[NVARS], vnid[NVARS];
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
        // enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
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
    a_real Rijd[NVARS], vxijd[NVARS], vyijd[NVARS], Hijd[NVARS], cijd[NVARS], vm2ijd[NVARS], vnijd[NVARS];
    arg1 = ur[0]/ul[0];
    Rij = sqrt(arg1);
    vxij = (Rij*vxj+vxi)/(Rij+1.0);
    vyij = (Rij*vyj+vyi)/(Rij+1.0);
    for (nd = 0; nd < NVARS; ++nd) {
        Rijd[nd] = (arg1 == 0.0 ? 0.0 : arg1d[nd]/(2.0*sqrt(arg1)));
        vxijd[nd] = ((vxj*Rijd[nd]+vxid[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((vyj*Rijd[nd]+vyid[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
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
        fluxd[0*NVARS+nd] = vnj*ur[0]*t1d[nd] + (t2d[nd]*vni+t2*vnid[nd])*ul[0] + t2*vni*uld[0][nd] - t3d[nd]*(ur[0]-ul[0]) + t3*uld[0][nd];
    }
    flux[0] = t1*vnj*ur[0] + t2*vni*ul[0] - t3*(ur[0]-ul[0]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[1*NVARS+nd] = (vnj*ur[1]+pj*n[0])*t1d[nd] + t2d[nd]*(vni*ul[1]+pi*n[0]) + t2*(vnid[nd]*ul[1]+vni*uld[1][nd]+n[0]*pid[nd]) 
			- t3d[nd]*(ur[1]-ul[1]) + t3*uld[1][nd];
    flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[2*NVARS+nd] = (vnj*ur[2]+pj*n[1])*t1d[nd] + t2d[nd]*(vni*ul[2]+pi*n[1]) + t2*(vnid[nd]*ul[2]+vni*uld[2][nd]+n[1]*pid[nd]) 
			- t3d[nd]*(ur[2]-ul[2]) + t3*uld[2][nd];
    flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[3*NVARS+nd] = vnj*ur[0]*Hj*t1d[nd] + (t2d[nd]*vni+t2*vnid[nd])*ul[0]*Hi + t2*vni*(uld[0][nd]*Hi+ul[0]*Hid[nd]) 
			- t3d[nd]*(ur[3]-ul[3]) + t3*uld[3][nd];
    flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi) - t3*(ur[3]-ul[3]);
}

/** Automatically differentiated Jacobian w.r.t. right state, generated by Tapenade 3.12 (r6213) - 13 Oct 2016 10:54.
 * Modified to remove the runtime parameter nbdirs and the differential of ul. Also changed the array shape of Jacobian.
 */
void HLLFlux::getFluxJac_right(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, const a_real *const __restrict__ n, 
		a_real *const __restrict__ flux, a_real *const __restrict__ fluxd)
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
    // enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
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
        vxijd[nd] = ((Rijd[nd]*vxj+Rij*vxjd[nd])*(Rij+1.0)-(Rij*vxj+vxi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
        vyijd[nd] = ((Rijd[nd]*vyj+Rij*vyjd[nd])*(Rij+1.0)-(Rij*vyj+vyi)*Rijd[nd])/((Rij+1.0)*(Rij+1.0));
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
        fluxd[1*NVARS+nd] = t1d[nd]*(vnj*ur[1]+pj*n[0]) + t1*(vnjd[nd]*ur[1]+vnj*urd[1][nd]+n[0]*pjd[nd]) + (vni*ul[1]+pi*n[0])*t2d[nd] 
			- t3d[nd]*(ur[1]-ul[1]) - t3*urd[1][nd];
    flux[1] = t1*(vnj*ur[1]+pj*n[0]) + t2*(vni*ul[1]+pi*n[0]) - t3*(ur[1]-ul[1]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[2*NVARS+nd] = t1d[nd]*(vnj*ur[2]+pj*n[1]) + t1*(vnjd[nd]*ur[2]+vnj*urd[2][nd]+n[1]*pjd[nd]) + (vni*ul[2]+pi*n[1])*t2d[nd] 
			- t3d[nd]*(ur[2]-ul[2]) - t3*urd[2][nd];
    flux[2] = t1*(vnj*ur[2]+pj*n[1]) + t2*(vni*ul[2]+pi*n[1]) - t3*(ur[2]-ul[2]);
    for (nd = 0; nd < NVARS; ++nd)
        fluxd[3*NVARS+nd] = (t1d[nd]*vnj+t1*vnjd[nd])*ur[0]*Hj + t1*vnj*(urd[0][nd]*Hj+ur[0]*Hjd[nd]) + vni*ul[0]*Hi*t2d[nd] 
			- t3d[nd]*(ur[3]-ul[3]) - t3*urd[3][nd];
    flux[3] = t1*(vnj*ur[0]*Hj) + t2*(vni*ul[0]*Hi) - t3*(ur[3]-ul[3]);
}

void HLLFlux::get_jacobian(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, const a_real* const __restrict__ n, 
		a_real *const __restrict__ dfdl, a_real *const __restrict__ dfdr)
{
	a_real flux[NVARS];
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

void HLLFlux::get_flux_jacobian(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, const a_real* const __restrict__ n, 
		a_real *const __restrict__ flux, a_real *const __restrict__ dfdl, a_real *const __restrict__ dfdr)
{
	getFluxJac_left(ul, ur, n, flux, dfdl);
	getFluxJac_right(ul, ur, n, flux, dfdr);
	for(int i = 0; i < NVARS*NVARS; i++)
		dfdl[i] *= -1.0;
}

/** The linearization assumes `locally frozen' signal speeds. According to Batten, Lechziner and Goldberg, this should be fine.
 */
void HLLFlux::get_frozen_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr)
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
	// enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
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
	aflux->evaluate_normal_jacobian(ul, n, dfdl);
	aflux->evaluate_normal_jacobian(ur, n, dfdr);
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

HLLCFlux::HLLCFlux(const a_real gamma, const EulerFlux *const analyticalflux) : InviscidFlux(gamma, analyticalflux)
{
}

/** Currently, the estimated signal speeds are the Einfeldt estimates, not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::get_flux(const a_real *const __restrict__ ul, const a_real *const __restrict__ ur, const a_real* const __restrict__ n, a_real *const __restrict__ flux)
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
	// enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
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
	const a_real sm = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj ) / ( ur[0]*(sr-vnj) - ul[0]*(sl-vni) );

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

void HLLCFlux::get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr)
{
	std::cout << " ! HLLCFlux: Not implemented!\n";
}

} // end namespace acfd

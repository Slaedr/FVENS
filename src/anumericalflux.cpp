/** \file anumericalflux.cpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#include <anumericalflux.hpp>

namespace acfd {

InviscidFlux::InviscidFlux(int num_vars, int num_dims, acfd_real gamma) : nvars(num_vars), ndim(num_dims), g(gamma)
{ }

InviscidFlux::~InviscidFlux()
{ }


VanLeerFlux::VanLeerFlux(int num_vars, int num_dims, acfd_real gamma) : InviscidFlux(num_vars, num_dims, gamma)
{
}

void VanLeerFlux::get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux)
{
	acfd_real nx, ny, pi, ci, vni, Mni, pj, cj, vnj, Mnj, vmags;
	acfd_real fiplus[NVARS], fjminus[NVARS];

	nx = n[0];
	ny = n[1];

	//calculate presures from u
	pi = (g-1)*(ul[3] - 0.5*(pow(ul[1],2)+pow(ul[2],2))/ul[0]);
	pj = (g-1)*(ur[3] - 0.5*(pow(ur[1],2)+pow(ur[2],2))/ur[0]);
	//calculate speeds of sound
	ci = sqrt(g*pi/ul[0]);
	cj = sqrt(g*pj/ur[0]);
	//calculate normal velocities
	vni = (ul[1]*nx +ul[2]*ny)/ul[0];
	vnj = (ur[1]*nx + ur[2]*ny)/ur[0];

	//Normal mach numbers
	Mni = vni/ci;
	Mnj = vnj/cj;

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
		vmags = pow(ul[1]/ul[0], 2) + pow(ul[2]/ul[0], 2);	// square of velocity magnitude
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
		vmags = pow(ur[1]/ur[0], 2) + pow(ur[2]/ur[0], 2);	// square of velocity magnitude
		fjminus[0] = -ur[0]*cj*pow(Mnj-1, 2)/4.0;
		fjminus[1] = fjminus[0] * (ur[1]/ur[0] + nx*(-2.0*cj - vnj)/g);
		fjminus[2] = fjminus[0] * (ur[2]/ur[0] + ny*(-2.0*cj - vnj)/g);
		fjminus[3] = fjminus[0] * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
	}

	//Update the flux vector
	for(int i = 0; i < NVARS; i++)
		flux[i] = fiplus[i] + fjminus[i];
}


RoeFlux::RoeFlux(int num_vars, int num_dims, acfd_real gamma) : InviscidFlux(num_vars, num_dims, gamma)
{ }

void RoeFlux::get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux)
{
	acfd_real Hi, Hj, ci, cj, pi, pj, vxi, vxj, vyi, vyj, vmag2i, vmag2j, vni, vnj;
	int ivar;

	vxi = ul[1]/ul[0]; vyi = ul[2]/ul[0];
	vxj = ur[1]/ur[0]; vyj = ur[2]/ur[0];
	vni = vxi*n[0] + vyi*n[1];
	vnj = vxj*n[0] + vyj*n[1];
	vmag2i = vxi*vxi + vyi*vyi;
	vmag2j = vxj*vxj + vyj*vyj;
	// pressures
	pi = (g-1.0)*(ul[3] - 0.5*ul[0]*vmag2i);
	pj = (g-1.0)*(ur[3] - 0.5*ur[0]*vmag2j);
	// speeds of sound
	ci = sqrt(g*pi/ul[0]);
	cj = sqrt(g*pj/ur[0]);
	// enthalpies  ( NOT E + p/rho = u(3)/u(0) + p/u(0) )
	Hi = g/(g-1.0)* pi/ul[0] + 0.5*vmag2i;
	Hj = g/(g-1.0)* pj/ur[0] + 0.5*vmag2j;

	// compute Roe-averages
	
	acfd_real Rij, rhoij, vxij, vyij, Hij, cij, vm2ij, vnij;
	Rij = sqrt(ur[0]/ul[0]);
	rhoij = Rij*ul[0];
	vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	vm2ij = vxij*vxij + vyij*vyij;
	vnij = vxij*n[0] + vyij*n[1];
	cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	// eigenvalues
	acfd_real l[4];
	l[0] = vnij; l[1] = vnij; l[2] = vnij + cij; l[3] = vnij - cij;

	// Harten-Hyman entropy fix
	acfd_real eps = 0;
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
	acfd_real r[4][4];
	
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
	
	for(ivar = 0; ivar < 4; ivar++)
	{
		r[ivar][2] *= rhoij/(2.0*cij);
		r[ivar][3] *= rhoij/(2.0*cij);
	}

	// R^(-1)(qR-qL)
	acfd_real dw[4];
	dw[0] = (ur[0]-ul[0]) - (pj-pi)/(cij*cij);
	dw[1] = (vxj-vxi)*n[1] - (vyj-vyi)*n[0];			// Dr Luo
	//dw(1) = rhoij;										// hack for conformance with Fink
	dw[2] = vnj-vni + (pj-pi)/(rhoij*cij);
	dw[3] = -(vnj-vni) + (pj-pi)/(rhoij*cij);

	// get one-sided flux vectors
	acfd_real fi[4], fj[4];
	fi[0] = ul[0]*vni;						fj[0] = ur[0]*vnj;
	fi[1] = ul[0]*vni*vxi + pi*n[0];		fj[1] = ur[0]*vnj*vxj + pj*n[0];
	fj[2] = ul[0]*vni*vyi + pi*n[1];		fj[2] = ur[0]*vnj*vyj + pj*n[1];
	fj[3] = vni*(ul[3] + pi);				fj[3] = vnj*(ur[3] + pj);

	// finally compute fluxes
	acfd_real sum; int j;
	for(ivar = 0; ivar < NVARS; ivar++)
	{
		sum = 0;
		for(j = 0; j < NVARS; j++)
			sum += fabs(l[j])*dw[j]*r[ivar][j];
		flux[ivar] = 0.5*(fi[ivar]+fj[ivar] - sum);
	}
}

HLLCFlux::HLLCFlux(int num_vars, int num_dims, acfd_real gamma) : InviscidFlux(num_vars, num_dims, gamma)
{
}

/** Currently, the estimated signal speeds are the classical estimates, not the corrected ones given by Remaki et. al.
 */
void HLLCFlux::get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux)
{
	acfd_real Hi, Hj, ci, cj, pi, pj, vxi, vxj, vyi, vyj, vmag2i, vmag2j, vni, vnj, pstar;
	acfd_real utemp[NVARS];
	int ivar;

	vxi = ul[1]/ul[0]; vyi = ul[2]/ul[0];
	vxj = ur[1]/ur[0]; vyj = ur[2]/ur[0];
	vni = vxi*n[0] + vyi*n[1];
	vnj = vxj*n[0] + vyj*n[1];
	vmag2i = vxi*vxi + vyi*vyi;
	vmag2j = vxj*vxj + vyj*vyj;
	// pressures
	pi = (g-1.0)*(ul[3] - 0.5*ul[0]*vmag2i);
	pj = (g-1.0)*(ur[3] - 0.5*ur[0]*vmag2j);
	// speeds of sound
	ci = sqrt(g*pi/ul[0]);
	cj = sqrt(g*pj/ur[0]);
	// enthalpies (E + p/rho = u(3)/u(0) + p/u(0) (actually specific enthalpy := enthalpy per unit mass)
	Hi = (ul[3] + pi)/ul[0];
	Hj = (ur[3] + pj)/ur[0];

	// compute Roe-averages
	acfd_real Rij, rhoij, vxij, vyij, Hij, cij, vm2ij, vnij;
	Rij = sqrt(ur[0]/ul[0]);
	rhoij = Rij*ul[0];
	vxij = (Rij*vxj + vxi)/(Rij + 1.0);
	vyij = (Rij*vyj + vyi)/(Rij + 1.0);
	Hij = (Rij*Hj + Hi)/(Rij + 1.0);
	vm2ij = vxij*vxij + vyij*vyij;
	vnij = vxij*n[0] + vyij*n[1];
	cij = sqrt( (g-1.0)*(Hij - vm2ij*0.5) );

	// estimate signal speeds (classical; not Remaki corrected)
	acfd_real sr, sl, sm;
	sl = vni - ci;
	if (sl > vnij-cij)
		sl = vnij-cij;
	sr = vnj+cj;
	if(sr < vnij+cij)
		sr = vnij+cij;
	sm = ( ur[0]*vnj*(sr-vnj) - ul[0]*vni*(sl-vni) + pi-pj ) / ( ur[0]*(sr-vnj) - ul[0]*(sl-vni) );

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

		pstar = ul[0]*(vni-sl)*(vni-sm) + pi;
		utemp[0] = ul[0] * (sl - vni)/(sl-sm);
		utemp[1] = ( (sl-vni)*ul[1] + (pstar-pi)*n[0] )/(sl-sm);
		utemp[2] = ( (sl-vni)*ul[2] + (pstar-pi)*n[1] )/(sl-sm);
		utemp[3] = ( (sl-vni)*ul[3] - pi*vni + pstar*sm )/(sl-sm);

		for(ivar = 0; ivar < NVARS; ivar++)
			flux[ivar] += sl * ( utemp[ivar] - ul[ivar]);
	}
	else if(sm <= 0 && sr >= 0)
	{
		flux[0] = vnj*ur[0];
		flux[1] = vnj*ur[1] + pj*n[0];
		flux[2] = vnj*ur[2] + pj*n[1];
		flux[3] = vnj*(ur[3] + pj);

		pstar = ur[0]*(vnj-sr)*(vnj-sm) + pj;
		utemp[0] = ur[0] * (sr - vnj)/(sr-sm);
		utemp[1] = ( (sr-vnj)*ur[1] + (pstar-pj)*n[0] )/(sr-sm);
		utemp[2] = ( (sr-vnj)*ur[2] + (pstar-pj)*n[1] )/(sr-sm);
		utemp[3] = ( (sr-vnj)*ur[3] - pj*vnj + pstar*sm )/(sr-sm);

		for(ivar = 0; ivar < NVARS; ivar++)
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

} // end namespace acfd

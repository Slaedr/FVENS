/** \file anumericalflux.hpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

//#include "aquadrature.hpp"
#include "alimiter.hpp"

#define __ANUMERICALFLUX_H 1

namespace acfd {

/// Adiabatic index
const double g = 1.4;

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, the numerical flux is computed.
 */
class InviscidFlux
{
protected:
	int nvars;		///< Number of conserved variables
	int ndim;		///< Number of spatial dimensions involved
	acfd_real g;	///< Adiabatic index
	/// Stores flux at each face
	amat::Matrix<acfd_real> fluxes;

public:
	/// Sets up data for the inviscid flux scheme
	InviscidFlux(int num_vars, int num_dims, acfd_real gamma);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const amat::Matrix<acfd_real>* const uleft, const amat::Matrix<acfd_real>* const uright, const acfd_real* const n, amat::Matrix<acfd_real>* const flux) = 0;
};

InviscidFlux::InviscidFlux(int num_vars, int num_dims, acfd_real gamma) : nvars(num_vars), ndim(num_dims), g(gamma)
{
	fluxes.setup(1,nvars);
}

/// Given left and right states at each face, the Van-Leer flux-vector-splitting is calculated at each face
class VanLeerFlux : public InviscidFlux
{
	amat::Matrix<double> fiplus;
	amat::Matrix<double> fjminus;

public:
	VanLeerFlux(int num_vars, int num_dims, acfd_real gamma);
	void get_flux(const amat::Matrix<acfd_real>* const ul, const amat::Matrix<acfd_real>* const ur, const acfd_real* const n, amat::Matrix<acfd_real>* const flux);
};

VanLeerFlux::VanLeerFlux(int num_vars, int num_dims, acfd_real gamma) : InviscidFlux(num_vars, num_dims, gamma)
{
	//InviscidFlux::setup(mesh, uleft, uright, rhs_ele, cfl_denom);
	fiplus.setup(nvars, 1);
	fjminus.setup(nvars, 1);
}

void VanLeerFlux::get_flux(const amat::Matrix<acfd_real>* ul, const amat::Matrix<acfd_real>* ur, const acfd_real* const n, amat::Matrix<acfd_real>* const flux)
{
	acfd_real nx, ny, pi, ci, vni, Mni, pj, cj, vnj, Mnj, vmags;

	nx = n[0];
	ny = n[1];

	//calculate presures from u
	pi = (g-1)*(ul->get(3) - 0.5*(pow(ul->get(1),2)+pow(ul->get(2),2))/ul->get(0));
	pj = (g-1)*(ur->get(3) - 0.5*(pow(ur->get(1),2)+pow(ur->get(2),2))/ur->get(0));
	//calculate speeds of sound
	ci = sqrt(g*pi/ul->get(0));
	cj = sqrt(g*pj/ur->get(0));
	//calculate normal velocities
	vni = (ul->get(1)*nx +ul->get(2)*ny)/ul->get(0);
	vnj = (ur->get(1)*nx + ur->get(2)*ny)/ur->get(0);

	//Normal mach numbers
	Mni = vni/ci;
	Mnj = vnj/cj;

	//Calculate split fluxes
	if(Mni < -1.0) fiplus.zeros();
	else if(Mni > 1.0)
	{
		fiplus(0) = ul->get(0)*vni;
		fiplus(1) = vni*ul->get(1) + pi*nx;
		fiplus(2) = vni*ul->get(2) + pi*ny;
		fiplus(3) = vni*(ul->get(3) + pi);
	}
	else
	{
		vmags = pow(ul->get(1)/ul->get(0), 2) + pow(ul->get(2)/ul->get(0), 2);	// square of velocity magnitude
		fiplus(0) = ul->get(0)*ci*pow(Mni+1, 2)/4.0;
		fiplus(1) = fiplus(0) * (ul->get(1)/ul->get(0) + nx*(2.0*ci - vni)/g);
		fiplus(2) = fiplus(0) * (ul->get(2)/ul->get(0) + ny*(2.0*ci - vni)/g);
		fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
	}

	if(Mnj > 1.0) fjminus.zeros();
	else if(Mnj < -1.0)
	{
		fjminus(0) = ur->get(0)*vnj;
		fjminus(1) = vnj*ur->get(1) + pj*nx;
		fjminus(2) = vnj*ur->get(2) + pj*ny;
		fjminus(3) = vnj*(ur->get(3) + pj);
	}
	else
	{
		vmags = pow(ur->get(1)/ur->get(0), 2) + pow(ur->get(2)/ur->get(0), 2);	// square of velocity magnitude
		fjminus(0) = -ur->get(0)*cj*pow(Mnj-1, 2)/4.0;
		fjminus(1) = fjminus(0) * (ur->get(1)/ur->get(0) + nx*(-2.0*cj - vnj)/g);
		fjminus(2) = fjminus(0) * (ur->get(2)/ur->get(0) + ny*(-2.0*cj - vnj)/g);
		fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
	}

	//Update the flux vector
	for(int i = 0; i < nvars; i++)
		(*flux)(i) = (fiplus(i) + fjminus(i));
}

} // end namespace acfd

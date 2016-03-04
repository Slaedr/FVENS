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
/** The class is such that given the left and right states at each face, fluxes are computed and added to the RHS of the left and right elements of each face.
 * The integral of the maximum magnitude of eigenvalue over each face is also computed:
 * \f[
 * \int_{f_i} (|v_n| + c) \mathrm{d}l
 * \f]
 * so that time steps can be calculated for explicit time stepping.
 * 
 * \note Currently implemented for one Gauss point per face
 */
class InviscidFlux
{
protected:
	const UTriMesh* m;
	int nvars;
	/// left-side state for each face
	const amat::Matrix<double>* ul;
	/// right-side state for each face
	const amat::Matrix<double>* ur;
	/// Stores total integral of flux for each element
	amat::Matrix<double>* rhsel;
	/// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each element, needed for computation of CFL number
	amat::Matrix<double>* cflden;
	/// Stores flux at each face
	amat::Matrix<double> fluxes;

public:
	/// Sets up data for the inviscid flux scheme
	/** \param[in] mesh is the mesh on which to compute fluxes
	 * \param[in] uleft is the vector of left states for each face (boundary and interior)
	 * \param[in] uright is the vector of right states for all faces
	 * \param rhs_ele is the vector containing the RHS for each element. The computed fluxes are summed into this vector.
	 * \param cfl_denom contains the denominator of the eigenvalue magnitude estimation for the purpose of computing CFL number.
	 */
	virtual void setup(const UTriMesh* mesh, const amat::Matrix<double>* uleft, const amat::Matrix<double>* uright, amat::Matrix<double>* rhs_ele, amat::Matrix<double>* cfl_denom);

	/// Actually carries out the flux computation
	virtual void compute_fluxes() = 0;
};

void InviscidFlux::setup(const UTriMesh* mesh, const amat::Matrix<double>* uleft, const amat::Matrix<double>* uright, amat::Matrix<double>* rhs_ele, amat::Matrix<double>* cfl_denom)
{
	m = mesh;
	ul = uleft;
	ur = uright;
	rhsel = rhs_ele;
	cflden = cfl_denom;
	nvars = ul->cols();			// we assume ul and ar are (naface x nvars) arrays
	fluxes.setup(m->gnaface(),nvars);
}

/// Given left and right states at each face, the Van-Leer flux-vector-splitting is calculated at each face
class VanLeerFlux : public InviscidFlux
{
	amat::Matrix<double> fiplus;
	amat::Matrix<double> fjminus;

public:
	void setup(const UTriMesh* mesh, const amat::Matrix<double>* uleft, const amat::Matrix<double>* uright, amat::Matrix<double>* rhs_ele, amat::Matrix<double>* cfl_denom);
	void compute_fluxes();
};

void VanLeerFlux::setup(const UTriMesh* mesh, const amat::Matrix<double>* uleft, const amat::Matrix<double>* uright, amat::Matrix<double>* rhs_ele, amat::Matrix<double>* cfl_denom)
{
	InviscidFlux::setup(mesh, uleft, uright, rhs_ele, cfl_denom);
	fiplus.setup(ul->cols(), 1);
	fjminus.setup(ul->cols(), 1);
}

void VanLeerFlux::compute_fluxes()
{
	acfd_real nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj, vmags;
	int lel, rel, ied;

	for(ied = 0; ied < m->gnaface(); ied++)
	{
		nx = m->ggallfa(ied,0);
		ny = m->ggallfa(ied,1);
		len = m->ggallfa(ied,2);
		lel = m->gintfac(ied,0);	// left element
		rel = m->gintfac(ied,1);	// right element

		//calculate presures from u
		pi = (g-1)*(ul->get(ied,3) - 0.5*(pow(ul->get(ied,1),2)+pow(ul->get(ied,2),2))/ul->get(ied,0));
		pj = (g-1)*(ur->get(ied,3) - 0.5*(pow(ur->get(ied,1),2)+pow(ur->get(ied,2),2))/ur->get(ied,0));
		//calculate speeds of sound
		ci = sqrt(g*pi/ul->get(ied,0));
		cj = sqrt(g*pj/ur->get(ied,0));
		//calculate normal velocities
		vni = (ul->get(ied,1)*nx +ul->get(ied,2)*ny)/ul->get(ied,0);
		vnj = (ur->get(ied,1)*nx + ur->get(ied,2)*ny)/ur->get(ied,0);

		//Normal mach numbers
		Mni = vni/ci;
		Mnj = vnj/cj;

		//Calculate split fluxes
		if(Mni < -1.0) fiplus.zeros();
		else if(Mni > 1.0)
		{
			fiplus(0) = ul->get(ied,0)*vni;
			fiplus(1) = vni*ul->get(ied,1) + pi*nx;
			fiplus(2) = vni*ul->get(ied,2) + pi*ny;
			fiplus(3) = vni*(ul->get(ied,3) + pi);
		}
		else
		{
			vmags = pow(ul->get(ied,1)/ul->get(ied,0), 2) + pow(ul->get(ied,2)/ul->get(ied,0), 2);	// square of velocity magnitude
			fiplus(0) = ul->get(ied,0)*ci*pow(Mni+1, 2)/4.0;
			fiplus(1) = fiplus(0) * (ul->get(ied,1)/ul->get(ied,0) + nx*(2.0*ci - vni)/g);
			fiplus(2) = fiplus(0) * (ul->get(ied,2)/ul->get(ied,0) + ny*(2.0*ci - vni)/g);
			fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
		}

		if(Mnj > 1.0) fjminus.zeros();
		else if(Mnj < -1.0)
		{
			fjminus(0) = ur->get(ied,0)*vnj;
			fjminus(1) = vnj*ur->get(ied,1) + pj*nx;
			fjminus(2) = vnj*ur->get(ied,2) + pj*ny;
			fjminus(3) = vnj*(ur->get(ied,3) + pj);
		}
		else
		{
			vmags = pow(ur->get(ied,1)/ur->get(ied,0), 2) + pow(ur->get(ied,2)/ur->get(ied,0), 2);	// square of velocity magnitude
			fjminus(0) = -ur->get(ied,0)*cj*pow(Mnj-1, 2)/4.0;
			fjminus(1) = fjminus(0) * (ur->get(ied,1)/ur->get(ied,0) + nx*(-2.0*cj - vnj)/g);
			fjminus(2) = fjminus(0) * (ur->get(ied,2)/ur->get(ied,0) + ny*(-2.0*cj - vnj)/g);
			fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
		}

		//Update the flux vector
		for(int i = 0; i < nvars; i++)
			fluxes(ied, i) = (fiplus(i) + fjminus(i));

		//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
		for(int i = 0; i < nvars; i++)
			fluxes(ied, i) *= len;

		// scatter the flux to elements' boundary integrands
		for(int i = 0; i < nvars; i++)
		{
			(*rhsel)(lel,i) -= fluxes(ied,i);
			if(rel >= 0 && rel < m->gnelem())
				(*rhsel)(rel,i) += fluxes(ied,i);
		}

		// calculate integ for CFL purposes
		//(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
		//(*cflden)(rel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
		(*cflden)(lel,0) += (dabs(vni) + ci)*len;
		if(rel >= 0 && rel < m->gnelem())
			(*cflden)(rel,0) += (dabs(vnj) + cj)*len;
	}
}

} // end namespace acfd

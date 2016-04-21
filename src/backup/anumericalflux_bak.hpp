/** \file anumericalflux.hpp
 * \brief Implements numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#ifndef __ACONSTANTS_H
#include <aconstants.h>
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
 */
class InviscidFlux
{
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
	 * \param[out] rhs_ele is the vector containing the RHS for each element. The computed fluxes are summed into this vector.
	 * \param[out] cfl_denom contains the denominator of the eigenvalue magnitude estimation for the purpose of computing CFL number.
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

VanLeerFlux::setup(const UTriMesh* mesh, const amat::Matrix<double>* uleft, const amat::Matrix<double>* uright, amat::Matrix<double>* rhs_ele, amat::Matrix<double>* cfl_denom)
{
	InviscidFlux::setup(mesh, uleft, uright, rhs_ele, cfl_denom);
	fiplus.setup(ul->cols(), 1);
	fjminus.setup(ul->cols(), 1);
}

void VanLeerFlux::compute_fluxes()
{
	double nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj, vmags;
	int lel, rel, ied;
	amat::Matrix<double> fiplus(u->cols(),1);
	amat::Matrix<double> fjminus(u->cols(),1);

	for(int ied = 0; ied < m->gnaface(); ied++)
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
		for(int i = 0; i < 4; i++)
			fluxes(ied, i) = (fiplus(i) + fjminus(i));

		//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
		for(int i = 0; i < nvars; i++)
			fluxes(ied, i) *= len;

		// scatter the flux to elements' boundary integrands
		for(int i = 0; i < nvars; i++)
		{
			//rhsel->operator()(lel,i) -= fluxes(ied,i);
			(*rhsel)(lel,i) -= fluxes(ied,i);
			//rhsel->operator()(rel,i) += fluxes(ied,i);
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

/// Inviscid flux calculation context for firs order Van-Leer flux-vector splitting
/** Asumption: boundary face flag is 2 for slip wall BC and 4 for inflow/outflow (far-field)
 */
class VanLeerFlux1
{
	UTriMesh* m;
	amat::Matrix<double>* u;				// nelem x nvars matrix holding values of unknowns at each cell
	amat::Matrix<double> fluxes;			// stores flux at each face
	amat::Matrix<double>* rhsel;
	amat::Matrix<double>* uinf;			// a state vector which has initial conditions
	amat::Matrix<double>* cflden;			// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each face
	int nvars;

public:
	VanLeerFlux1() {}

	VanLeerFlux1(UTriMesh* mesh, amat::Matrix<double>* unknowns, amat::Matrix<double>* uinit, amat::Matrix<double>* rhsel_in, amat::Matrix<double>* cfl_den)
	{
		nvars = unknowns->cols();
		m = mesh;
		u = unknowns;
		uinf = uinit;
		rhsel = rhsel_in;
		fluxes.setup(m->gnaface(), u->cols(), amat::ROWMAJOR);
	}

	void setup(UTriMesh* mesh, amat::Matrix<double>* unknowns, amat::Matrix<double>* uinit, amat::Matrix<double>* rhsel_in, amat::Matrix<double>* cfl_den)
	{
		nvars = unknowns->cols();
		m = mesh;
		u = unknowns;
		uinf = uinit;
		rhsel = rhsel_in;
		fluxes.setup(m->gnaface(), u->cols(), amat::ROWMAJOR);
		cflden = cfl_den;
	}

	void compute_fluxes()
	{
		//cout << "VanLeerFlux: compute_fluxes(): Computing fluxes at boundary faces\n";
		//loop over boundary faces. Need to account for boundary conditions here
		double nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj;
		int lel, rel;
		amat::Matrix<double> fiplus(u->cols(),1,amat::ROWMAJOR);
		amat::Matrix<double> fjminus(u->cols(),1,amat::ROWMAJOR);
		amat::Matrix<double> ug(1, u->cols(), amat::ROWMAJOR);		// stores ghost state

		for(int ied = 0; ied < m->gnbface(); ied++)
		{
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			//int rel = m->gintfac(ied,1);	// right element - does not exist

			pi = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
			ci = sqrt(g*pi/u->get(lel,0));
			vni = (u->get(lel,1)*nx + u->get(lel,2)*ny)/u->get(lel,0);
			Mni = vni/ci;

			//get boundary face flag for ied and set the ghost state
			if(m->ggallfa(ied,3) == 2)		// solid wall
			{
				ug(0,0) = u->get(lel,0);
				ug(0,1) = u->get(lel,1) - 2*vni*nx*ug(0,0);
				ug(0,2) = u->get(lel,2) - 2*vni*ny*ug(0,0);
				//ug(0,1) = u->get(lel,1)*(ny*ny-nx*nx) - u->get(lel,2)*ny*(1.0+nx);
				//ug(0,2) = -2.0*u->get(lel,1)*nx*ny + u->get(lel,2)*(nx*nx-ny*ny);
				ug(0,3) = u->get(lel,3);

				pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
				cj = sqrt(g*pj/ug(0,0));
				vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
			}
			if(m->ggallfa(ied,3) == 4)		// inflow or outflow DOES NOT WORK
			{
				/*if(Mni < -1.0)
				{
					for(int i = 0; i < nvars; i++)
						ug(0,i) = uinf->get(0,i);
					pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
					cj = sqrt(g*pj/ug(0,0));
					vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
				}
				else if(Mni >= -1.0 && Mni < 0.0)
				{
					double vinfx = uinf->get(0,1)/uinf->get(0,0);
					double vinfy = uinf->get(0,2)/uinf->get(0,0);
					double vinfn = vinfx*nx + vinfy*ny;
					double vbn = u->get(lel,1)/u->get(lel,0)*nx + u->get(lel,2)/u->get(lel,0)*ny;
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/u->get(lel,0));

					double vgx = vinfx*ny*ny - vinfy*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vinfy*nx*nx - vinfx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ug(0,0) = pow( pinf/pow(uinf->get(0,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ug(0,0)/g*cj*cj;

					ug(0,3) = pj/(g-1) + 0.5*ug(0,0)*(vgx*vgx+vgy*vgy);
					ug(0,1) = ug(0,0)*vgx;
					ug(0,2) = ug(0,0)*vgy;
				}
				else if(Mni >= 0.0 && Mni < 1.0)
				{
					double vbx = u->get(lel,1)/u->get(lel,0);
					double vby = u->get(lel,2)/u->get(lel,0);
					double vbn = vbx*nx + vby*ny;
					double vinfn = uinf->get(0,1)/uinf->get(0,0)*nx + uinf->get(0,2)/uinf->get(0,0)*ny;
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/u->get(lel,0));

					double vgx = vbx*ny*ny - vby*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vby*nx*nx - vbx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ug(0,0) = pow( pb/pow(u->get(lel,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ug(0,0)/g*cj*cj;

					ug(0,3) = pj/(g-1) + 0.5*ug(0,0)*(vgx*vgx+vgy*vgy);
					ug(0,1) = ug(0,0)*vgx;
					ug(0,2) = ug(0,0)*vgy;
				}
				else
				{
					for(int i = 0; i < nvars; i++)
						ug(0,i) = u->get(lel,i);
					pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
					cj = sqrt(g*pj/ug(0,0));
					vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
				}*/

				// Naive way
				for(int i = 0; i < nvars; i++)
					ug(0,i) = uinf->get(0,i);
				pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
				cj = sqrt(g*pj/ug(0,0));
				vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
			}

			//calculate other flow variables at ghost state
			//double pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
			//double cj = sqrt(g*pj/ug(0,0));
			//double vnj = ug(0,1)*nx + ug(0,2)*ny;
			Mnj = vnj/cj;

			//cout << " " << cj;

			//calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = u->get(lel,0)*vni;
				fiplus(1) = vni*u->get(lel,1) + pi*nx;
				fiplus(2) = vni*u->get(lel,2) + pi*ny;
				fiplus(3) = vni*(u->get(lel,3) + pi);
			}
			else
			{
				double vmags = pow(u->get(lel,1)/u->get(lel,0), 2) + pow(u->get(lel,2)/u->get(lel,0), 2);	// square of velocity magnitude
				fiplus(0) = u->get(lel,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * (u->get(lel,1)/u->get(lel,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * (u->get(lel,2)/u->get(lel,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = ug(0,0)*vnj;
				fjminus(1) = vnj*ug(0,1) + pj*nx;
				fjminus(2) = vnj*ug(0,2) + pj*ny;
				fjminus(3) = vnj*(ug(0,3) + pj);
			}
			else
			{
				double vmags = pow(ug(0,1)/ug(0,0), 2) + pow(ug(0,2)/ug(0,0), 2);	// square of velocity magnitude
				fjminus(0) = -ug(0,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * (ug(0,1)/ug(0,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * (ug(0,2)/ug(0,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) = (fiplus(i) + fjminus(i));

			//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) *= len;

			// scatter the flux to elements' boundary integrals
			for(int i = 0; i < nvars; i++)
				//rhsel->operator()(lel,i) -= fluxes(ied,i);
				(*rhsel)(lel,i) -= fluxes(ied,i);

			// calculate integ for CFL purposes
			(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
		}

		//cout << "VanLeerFlux: compute_fluxes(): Computing fluxes at internal faces\n";
		//loop over internal faces
		for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			rel = m->gintfac(ied,1);	// right element

			//calculate presures from u
			pi = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
			pj = (g-1)*(u->get(rel,3) - 0.5*(pow(u->get(rel,1),2)+pow(u->get(rel,2),2))/u->get(rel,0));
			//calculate speeds of sound
			ci = sqrt(g*pi/u->get(lel,0));
			cj = sqrt(g*pj/u->get(rel,0));
			//calculate normal velocities
			vni = (u->get(lel,1)*nx + u->get(lel,2)*ny)/u->get(lel,0);
			vnj = (u->get(rel,1)*nx + u->get(rel,2)*ny)/u->get(rel,0);

			//Normal mach numbers
			Mni = vni/ci;
			Mnj = vnj/cj;

			//Calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = u->get(lel,0)*vni;
				fiplus(1) = vni*u->get(lel,1) + pi*nx;
				fiplus(2) = vni*u->get(lel,2) + pi*ny;
				fiplus(3) = vni*(u->get(lel,3) + pi);
			}
			else
			{
				double vmags = pow(u->get(lel,1)/u->get(lel,0), 2) + pow(u->get(lel,2)/u->get(lel,0), 2);	// square of velocity magnitude
				fiplus(0) = u->get(lel,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * (u->get(lel,1)/u->get(lel,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * (u->get(lel,2)/u->get(lel,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = u->get(rel,0)*vnj;
				fjminus(1) = vnj*u->get(rel,1) + pj*nx;
				fjminus(2) = vnj*u->get(rel,2) + pj*ny;
				fjminus(3) = vnj*(u->get(rel,3) + pj);
			}
			else
			{
				double vmags = pow(u->get(rel,1)/u->get(rel,0), 2) + pow(u->get(rel,2)/u->get(rel,0), 2);	// square of velocity magnitude
				fjminus(0) = -u->get(rel,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * (u->get(rel,1)/u->get(rel,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * (u->get(rel,2)/u->get(rel,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < 4; i++)
				fluxes(ied, i) = (fiplus(i) + fjminus(i));

			//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) *= len;

			// scatter the flux to elements' boundary integrands
			for(int i = 0; i < nvars; i++)
			{
				//rhsel->operator()(lel,i) -= fluxes(ied,i);
				(*rhsel)(lel,i) -= fluxes(ied,i);
				//rhsel->operator()(rel,i) += fluxes(ied,i);
				(*rhsel)(rel,i) += fluxes(ied,i);
			}

			// calculate integ for CFL purposes
			(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
			(*cflden)(rel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
		}
		//cout << "VanLeerFlux: compute_fluxes(): Fluxes computed.\n";
	}

};


typedef VanAlbadaLimiter Limiter;

/// Van Leer fluxes with MUSCL reconstruction for 2nd order accuracy
class VanLeerFlux2
{
	UTriMesh* m;
	amat::Matrix<double>* u;				// nelem x nvars matrix holding values of unknowns at each cell
	amat::Matrix<double> fluxes;			// stores flux at each face
	amat::Matrix<double>* rhsel;
	amat::Matrix<double>* uinf;			// a state vector which has initial conditions
	amat::Matrix<double> ug;				// holds ghost states for each boundary face

	/// stores whether the limiter context has been allocated
	bool limiter_allocated;

	/// stores \f$ int_{\partial \Omega_I} ( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face
	amat::Matrix<double>* cflden;

	int nvars;

	/// x centroids of elements
	amat::Matrix<double> xi;
	/// y centroids of elements
	amat::Matrix<double> yi;
	/// x centroids of ghost elements
	amat::Matrix<double> xb;
	/// y centroids of ghost elements
	amat::Matrix<double> yb;

	/// number of gauss points per face
	int ngauss;
	/// holds x-coordinates of gauss points
	amat::Matrix<double> gaussx;
	/// holds y-coordinates of gauss points
	amat::Matrix<double> gaussy;

	/// x-components of gradients of flow (conserved) variables
	amat::Matrix<double>* dudx;
	/// y-components of gradients of flow (conserved) variables
	amat::Matrix<double>* dudy;
	/// LHS coeffs for least-squares reconstruction
	amat::Matrix<double>* w2x2;
	/// LHS coeffs for least-squares reconstruction
	amat::Matrix<double>* w2y2;
	/// LHS coeffs for least-squares reconstruction
	amat::Matrix<double>* w2xy;

	/// left data (of conserved variables) at each face
	amat::Matrix<double> ufl;
	/// right data (of conserved variables) at each face
	amat::Matrix<double> ufr;

	/// Limiter context
	Limiter* lim;
	/// value of limiter \f$ \phi \f$ used for computing limited flow variables of left cell
	amat::Matrix<double> phi_l;
	/// value of limiter \f$ \phi \f$ used for computing limited flow variables of right cell
	amat::Matrix<double> phi_r;

public:
	VanLeerFlux2()
	{
		limiter_allocated = false;
	}
	void setup(UTriMesh* mesh, amat::Matrix<double>* unknowns, amat::Matrix<double>* derx, amat::Matrix<double>* dery, amat::Matrix<double>* _w2x2, amat::Matrix<double>* _w2y2, amat::Matrix<double>* _w2xy, amat::Matrix<double>* uinit, amat::Matrix<double>* rhsel_in, amat::Matrix<double>* cfl_den, int gauss_order)
	{
		nvars = unknowns->cols();
		m = mesh;
		u = unknowns;
		dudx = derx;
		dudy = dery;
		w2x2 = _w2x2;
		w2y2 = _w2y2;
		w2xy = _w2xy;
		uinf = uinit;
		rhsel = rhsel_in;
		fluxes.setup(m->gnaface(), u->cols(), amat::ROWMAJOR);
		ufl.setup(m->gnaface(), u->cols(), amat::ROWMAJOR);
		ufr.setup(m->gnaface(), u->cols(), amat::ROWMAJOR);
		ug.setup(m->gnbface(),u->cols(),amat::ROWMAJOR);
		cflden = cfl_den;
		ngauss = gauss_order;
		gaussx.setup(m->gnaface(),ngauss,amat::ROWMAJOR);
		gaussy.setup(m->gnaface(),ngauss,amat::ROWMAJOR);
		xi.setup(m->gnelem(),1,amat::ROWMAJOR);
		yi.setup(m->gnelem(),1,amat::ROWMAJOR);
		xb.setup(m->gnbface(),1,amat::ROWMAJOR);
		yb.setup(m->gnbface(),1,amat::ROWMAJOR);
		phi_l.setup(m->gnaface(), nvars, amat::ROWMAJOR);
		phi_r.setup(m->gnaface(), nvars, amat::ROWMAJOR);

		lim = new Limiter();
		limiter_allocated = true;
		//lim.setup_limiter(m, u, &ug, &xb, &yb, &xi, &yi, &phi);

		// Calculate centroids of elements and coordinates of gauss points
		// Boundary faces
		//cout << "VanLeerFlux2: setup(): Calculating centroids around boundary faces\n";
		int ielem; double nx, ny, x1,x2,y1,y2, xs,ys;
		for(int ied = 0; ied < m->gnbface(); ied++)
		{
			ielem = m->gintfac(ied,0); //int lel = ielem;
			//int jelem = m->gintfac(ied,1); //int rel = jelem;
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);

			xi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi(ielem) /= 3.0;
			yi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi(ielem) /= 3.0;

			x1 = m->gcoords(m->gintfac(ied,2),0);
			x2 = m->gcoords(m->gintfac(ied,3),0);
			y1 = m->gcoords(m->gintfac(ied,2),1);
			y2 = m->gcoords(m->gintfac(ied,3),1);
			//double xs, ys;
			if(dabs(nx)>A_SMALL_NUMBER && dabs(ny)>A_SMALL_NUMBER)  // check if nx = 0
			{
				//xs = (y2-y1)/(x2-x1)*x1 + ny/nx*xi(ielem) + y1-yi(ielem);
				xs = ( yi(ielem)-y1 - ny/nx*xi(ielem) + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
				ys = ny/nx*xs + yi(ielem) - ny/nx*xi(ielem);
			}
			else if(dabs(nx)<=A_SMALL_NUMBER)
			{
				xs = xi(ielem);
				ys = y1;
			}
			else
			{
				xs = x1;
				ys = yi(ielem);
			}
			xb(ied) = 2*xs-xi(ielem);
			yb(ied) = 2*ys-yi(ielem);
		}
		//xb.mprint(); yb.mprint();

		//cout << "VanLeerFlux2: setup(): Calculating centroids around internal faces\n";
		// Internal element centroids
		for(int ielem = 0; ielem < m->gnelem(); ielem++)
		{
			xi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi(ielem) = xi(ielem) / 3.0;
			yi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi(ielem) = yi(ielem) / 3.0;
		}
		//xi.mprint(); yi.mprint();

		//cout << "VanLeerFlux2: setup(): Calculating gauss points\n";
		//Calculate and store coordinates of GAUSS POINTS (general implementation)
		// Gauss points are uniformly distributed along the face.
		int ig;
		for(int ied = 0; ied < m->gnaface(); ied++)
		{
			x1 = m->gcoords(m->gintfac(ied,2),0);
			y1 = m->gcoords(m->gintfac(ied,2),1);
			x2 = m->gcoords(m->gintfac(ied,3),0);
			y2 = m->gcoords(m->gintfac(ied,3),1);
			for(ig = 0; ig < ngauss; ig++)
			{
				gaussx(ied,ig) = x1 + (double)(ig+1)/(double)(ngauss+1) * (x2-x1);
				gaussy(ied,ig) = y1 + (double)(ig+1)/(double)(ngauss+1) * (y2-y1);
			}
		}
		//gaussx.mprint(); gaussy.mprint();
		//cout << "VanLeerFlux2: setup(): Done\n";
	}

	~VanLeerFlux2()
	{
		if(limiter_allocated)
			delete lim;
	}

	/// Computes flow variables in ghost cells using characteristic boundary conditions
	/** Required for computing RHS of least-squares problem.
	 */
	void compute_ghostcell_states()
	{
		//cout << "VanLeerFlux2: compute_ghostcell_states()\n";
		int lel, rel;
		double nx, ny, vni, pi, ci, Mni, vnj, pj, cj, Mnj;
		for(int ied = 0; ied < m->gnbface(); ied++)
		{
			lel = m->gintfac(ied,0);
			//int jelem = m->gintfac(ied,1);
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);

			vni = (u->get(lel,1)*nx + u->get(lel,2)*ny)/u->get(lel,0);
			pi = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
			ci = sqrt(g*pi/u->get(lel,0));
			Mni = vni/ci;

			if(m->ggallfa(ied,3) == 2)		// solid wall
			{
				ug(ied,0) = u->get(lel,0);
				ug(ied,1) = u->get(lel,1) - 2*vni*nx*ug(ied,0);
				ug(ied,2) = u->get(lel,2) - 2*vni*ny*ug(ied,0);
				ug(ied,3) = u->get(lel,3);
			}

			if(m->ggallfa(ied,3) == 4)		// inflow or outflow
			{
				/*if(Mni < -1.0)
				{
					for(int i = 0; i < nvars; i++)
						ug(0,i) = uinf->get(0,i);
					pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
					cj = sqrt(g*pj/ug(0,0));
					vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
				}
				else if(Mni >= -1.0 && Mni < 0.0)
				{
					double vinfx = uinf->get(0,1)/uinf->get(0,0);
					double vinfy = uinf->get(0,2)/uinf->get(0,0);
					double vinfn = vinfx*nx + vinfy*ny;
					double vbn = u->get(lel,1)/u->get(lel,0)*nx + u->get(lel,2)/u->get(lel,0)*ny;
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/u->get(lel,0));

					double vgx = vinfx*ny*ny - vinfy*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vinfy*nx*nx - vinfx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ug(0,0) = pow( pinf/pow(uinf->get(0,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ug(0,0)/g*cj*cj;

					ug(0,3) = pj/(g-1) + 0.5*ug(0,0)*(vgx*vgx+vgy*vgy);
					ug(0,1) = ug(0,0)*vgx;
					ug(0,2) = ug(0,0)*vgy;
				}
				else if(Mni >= 0.0 && Mni < 1.0)
				{
					double vbx = u->get(lel,1)/u->get(lel,0);
					double vby = u->get(lel,2)/u->get(lel,0);
					double vbn = vbx*nx + vby*ny;
					double vinfn = uinf->get(0,1)/uinf->get(0,0)*nx + uinf->get(0,2)/uinf->get(0,0)*ny;
					double pinf = (g-1)*(uinf->get(0,3) - 0.5*(pow(uinf->get(0,1),2)+pow(uinf->get(0,2),2))/uinf->get(0,0));
					double pb = (g-1)*(u->get(lel,3) - 0.5*(pow(u->get(lel,1),2)+pow(u->get(lel,2),2))/u->get(lel,0));
					double cinf = sqrt(g*pinf/uinf->get(0,0));
					double cb = sqrt(g*pb/u->get(lel,0));

					double vgx = vbx*ny*ny - vby*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
					double vgy = vby*nx*nx - vbx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
					vnj = vgx*nx + vgy*ny;	// = vgn
					cj = (g-1)/2*(vnj-vinfn)+cinf;
					ug(0,0) = pow( pb/pow(u->get(lel,0),g) * 1.0/cj*cj , 1/(1-g));	// density
					pj = ug(0,0)/g*cj*cj;

					ug(0,3) = pj/(g-1) + 0.5*ug(0,0)*(vgx*vgx+vgy*vgy);
					ug(0,1) = ug(0,0)*vgx;
					ug(0,2) = ug(0,0)*vgy;
				}
				else
				{
					for(int i = 0; i < nvars; i++)
						ug(0,i) = u->get(lel,i);
					pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
					cj = sqrt(g*pj/ug(0,0));
					vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
				} */

				// Naive way
				for(int i = 0; i < nvars; i++)
					ug(ied,i) = uinf->get(0,i);
			}
		}
		//cout << "VanLeerFlux2: compute_ghostcell_states(): Done\n";
		//ug.mprint();
	}

	/// Calculate gradients by Least-squares reconstruction
	void compute_ls_reconstruction()
	{
		//cout << "VanLeerFlux2: compute_ls_gradients(): Computing ghost cell data...\n";
		amat::Matrix<double> w2yu(m->gnelem(),nvars), w2xu(m->gnelem(),nvars);
		w2xu.zeros(); w2yu.zeros();

		//calculate ghost states
		compute_ghostcell_states();

		// Boundary faces
		int ielem; double nx, ny;
		for(int ied = 0; ied < m->gnbface(); ied++)
		{
			ielem = m->gintfac(ied,0); //int lel = ielem;
			//int jelem = m->gintfac(ied,1); //int rel = jelem;
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);

			for(int i = 0; i < nvars; i++)
			{
				w2xu(ielem,i) += 1.0/( (xb(ied)-xi(ielem))*(xb(ied)-xi(ielem))+(yb(ied)-yi(ielem))*(yb(ied)-yi(ielem)) )*(xb(ied)-xi(ielem))*(ug.get(ied,i) - u->get(ielem,i));
				w2yu(ielem,i) += 1.0/( (xb(ied)-xi(ielem))*(xb(ied)-xi(ielem))+(yb(ied)-yi(ielem))*(yb(ied)-yi(ielem)) )*(yb(ied)-yi(ielem))*(ug.get(ied,i) - u->get(ielem,i));
			}
		}

		// Internal faces
		int jelem, i;
		for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			jelem = m->gintfac(ied,1);

			for(i = 0; i < nvars; i++)
			{
				w2xu(ielem,i) += 1.0/( (xi(jelem)-xi(ielem))*(xi(jelem)-xi(ielem))+(yi(jelem)-yi(ielem))*(yi(jelem)-yi(ielem)) )*(xi(jelem)-xi(ielem))*(u->get(jelem,i) - u->get(ielem,i));
				w2yu(ielem,i) += 1.0/( (xi(jelem)-xi(ielem))*(xi(jelem)-xi(ielem))+(yi(jelem)-yi(ielem))*(yi(jelem)-yi(ielem)) )*(yi(jelem)-yi(ielem))*(u->get(jelem,i) - u->get(ielem,i));

				w2xu(jelem,i) += 1.0/( (xi(jelem)-xi(ielem))*(xi(jelem)-xi(ielem))+(yi(jelem)-yi(ielem))*(yi(jelem)-yi(ielem)) )*(xi(ielem)-xi(jelem))*(u->get(ielem,i) - u->get(jelem,i));
				w2yu(jelem,i) += 1.0/( (xi(jelem)-xi(ielem))*(xi(jelem)-xi(ielem))+(yi(jelem)-yi(ielem))*(yi(jelem)-yi(ielem)) )*(yi(ielem)-yi(jelem))*(u->get(ielem,i) - u->get(jelem,i));
			}
		}
		//w2xu.mprint();

		// calculate gradient
		//cout << "VanLeerFlux2: compute_ls_gradients(): Computing gradients\n";
		for(int ielem = 0; ielem < m->gnelem(); ielem++)
			for(int i = 0; i < nvars; i++)
			{
				(*dudy)(ielem,i) = (w2yu(ielem,i)*(*w2x2)(ielem) - w2xu(ielem,i)*(*w2xy)(ielem)) / ((*w2y2)(ielem)*(*w2x2)(ielem)-(*w2xy)(ielem)*(*w2xy)(ielem));
				(*dudx)(ielem,i) =  w2xu(ielem,i)/(*w2x2)(ielem) - (*w2xy)(ielem)/(*w2x2)(ielem)*(*dudy)(ielem,i);
			}

		//compute limiters

		lim->setup_limiter(m, u, &ug, dudx, dudy, &xb, &yb, &xi, &yi, &gaussx, &gaussy, &phi_l, &phi_r, &ufl, &ufr, uinf, g);
		//lim->compute_limiters(); 		// ----------- enable for shock capturing! -----------
		//lim->compute_interface_values();
		lim->compute_unlimited_interface_values();

		// cout << "VanLeerFlux2: compute_ls_gradients(): Computing values at faces - internal\n";
	}


	/// Compute flux values at each face, integrate, and add to residual rhsel .
	void compute_fluxes()
	{
		//cout << "VanLeerFlux2: compute_fluxes(): Computing fluxes at boundary faces\n";

		//FIrst compute gradients and values at interfaces. This calculates ghost states as well (in ufr).
		compute_ls_reconstruction();

		double nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj;
		int lel, rel;
		amat::Matrix<double> fiplus(u->cols(),1,amat::ROWMAJOR);
		amat::Matrix<double> fjminus(u->cols(),1,amat::ROWMAJOR);
		int ied;

		//loop over boundary faces
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			//int rel = m->gintfac(ied,1);	// right element - does not exist

			pi = (g-1)*(ufl(ied,3) - 0.5*(pow(ufl(ied,1),2)+pow(ufl(ied,2),2))/ufl(ied,0));
			ci = sqrt(g*pi/ufl(ied,0));
			vni = (ufl(ied,1)*nx + ufl(ied,2)*ny)/ufl(ied,0);
			Mni = vni/ci;

			// other ghost state quantities
			pj = (g-1)*(ufr(ied,3) - 0.5*(pow(ufr(ied,1),2)+pow(ufr(ied,2),2))/ufr(ied,0));
			cj = sqrt(g*pj/ufr(ied,0));
			vnj = (ufr(ied,1)*nx + ufr(ied,2)*ny)/ufr(ied,0);
			Mnj = vnj/cj;

			//cout << " " << cj;

			//calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = ufl(ied,0)*vni;
				fiplus(1) = vni*ufl(ied,1) + pi*nx;
				fiplus(2) = vni*ufl(ied,2) + pi*ny;
				fiplus(3) = vni*(ufl(ied,3) + pi);
			}
			else
			{
				double vmags = pow(ufl(ied,1)/ufl(ied,0), 2) + pow(ufl(ied,2)/ufl(ied,0), 2);	// square of velocity magnitude
				fiplus(0) = ufl(ied,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * (ufl(ied,1)/ufl(ied,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * (ufl(ied,2)/ufl(ied,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = ufr(ied,0)*vnj;
				fjminus(1) = vnj*ufr(ied,1) + pj*nx;
				fjminus(2) = vnj*ufr(ied,2) + pj*ny;
				fjminus(3) = vnj*(ufr(ied,3) + pj);
			}
			else
			{
				double vmags = pow(ufr(ied,1)/ufr(ied,0), 2) + pow(ufr(ied,2)/ufr(ied,0), 2);	// square of velocity magnitude
				fjminus(0) = -ufr(ied,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * (ufr(ied,1)/ufr(ied,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * (ufr(ied,2)/ufr(ied,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) = (fiplus(i) + fjminus(i));

			//Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) *= len;

			// scatter the flux to elements' boundary integrals
			for(int i = 0; i < nvars; i++)
				//rhsel->operator()(lel,i) -= fluxes(ied,i);
				(*rhsel)(lel,i) -= fluxes(ied,i);

			// calculate integ for CFL purposes
			//(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
			(*cflden)(lel,0) += (dabs(vni) + ci)*len;
		}

		//cout << "VanLeerFlux2: compute_fluxes(): Computing fluxes at internal faces\n";
		//loop over internal faces
		for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			rel = m->gintfac(ied,1);	// right element

			amat::Matrix<double> fiplus(u->cols(),1,amat::ROWMAJOR);
			amat::Matrix<double> fjminus(u->cols(),1,amat::ROWMAJOR);

			//calculate presures from u
			double pi = (g-1)*(ufl(ied,3) - 0.5*(pow(ufl(ied,1),2)+pow(ufl(ied,2),2))/ufl(ied,0));
			double pj = (g-1)*(ufr(ied,3) - 0.5*(pow(ufr(ied,1),2)+pow(ufr(ied,2),2))/ufr(ied,0));
			//calculate speeds of sound
			double ci = sqrt(g*pi/ufl(ied,0));
			double cj = sqrt(g*pj/ufr(ied,0));
			//calculate normal velocities
			double vni = (ufl(ied,1)*nx +ufl(ied,2)*ny)/ufl(ied,0);
			double vnj = (ufr(ied,1)*nx + ufr(ied,2)*ny)/ufr(ied,0);

			//Normal mach numbers
			double Mni = vni/ci;
			double Mnj = vnj/cj;

			//Calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = ufl(ied,0)*vni;
				fiplus(1) = vni*ufl(ied,1) + pi*nx;
				fiplus(2) = vni*ufl(ied,2) + pi*ny;
				fiplus(3) = vni*(ufl(ied,3) + pi);
			}
			else
			{
				double vmags = pow(ufl(ied,1)/ufl(ied,0), 2) + pow(ufl(ied,2)/ufl(ied,0), 2);	// square of velocity magnitude
				fiplus(0) = ufl(ied,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * (ufl(ied,1)/ufl(ied,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * (ufl(ied,2)/ufl(ied,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = ufr(ied,0)*vnj;
				fjminus(1) = vnj*ufr(ied,1) + pj*nx;
				fjminus(2) = vnj*ufr(ied,2) + pj*ny;
				fjminus(3) = vnj*(ufr(ied,3) + pj);
			}
			else
			{
				double vmags = pow(ufr(ied,1)/ufr(ied,0), 2) + pow(ufr(ied,2)/ufr(ied,0), 2);	// square of velocity magnitude
				fjminus(0) = -ufr(ied,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * (ufr(ied,1)/ufr(ied,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * (ufr(ied,2)/ufr(ied,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < 4; i++)
				fluxes(ied, i) = (fiplus(i) + fjminus(i));

			//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(ied, i) *= len;

			// scatter the flux to elements' boundary integrands
			for(int i = 0; i < nvars; i++)
			{
				//rhsel->operator()(lel,i) -= fluxes(ied,i);
				(*rhsel)(lel,i) -= fluxes(ied,i);
				//rhsel->operator()(rel,i) += fluxes(ied,i);
				(*rhsel)(rel,i) += fluxes(ied,i);
			}

			// calculate integ for CFL purposes
			//(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
			//(*cflden)(rel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
			(*cflden)(lel,0) += (dabs(vni) + ci)*len;
			(*cflden)(rel,0) += (dabs(vnj) + cj)*len;
		}
		//cout << "VanLeerFlux2: compute_fluxes(): Fluxes computed.\n";
	}

	amat::Matrix<double> centroid_x_coords() {
		return xi;
	}
	amat::Matrix<double> centroid_y_coords() {
		return yi;
	}
};

} // end namespace acfd

#ifdef _OPENMP
#ifndef OMP_H
#include <omp.h>
#endif
#endif

#ifndef _GLIBCXX_CSTDIO
#include <cstdio>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#include "aquadrature.hpp"
#include "alimiter_parallel.hpp"

#ifndef NTHREADS
#define NTHREADS 2
#endif

using namespace amat;
using namespace acfd;

const int nthreads = 2;

namespace acfd {

const double g = 1.4;

//Asumption: boundary face flag is 2 for slip wall BC and 4 for inflow/outflow (far-field)
class VanLeerFlux
{
	UTriMesh* m;
	Matrix<double>* u;				// nelem x nvars matrix holding values of unknowns at each cell
	Matrix<double> fluxes;			// stores flux at each face
	Matrix<double>* rhsel;
	Matrix<double>* uinf;			// a state vector which has initial conditions
	Matrix<double>* cflden;			// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each face
	int nvars;

public:
	VanLeerFlux() {}

	VanLeerFlux(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* uinit, Matrix<double>* rhsel_in, Matrix<double>* cfl_den)
	{
		nvars = unknowns->cols();
		m = mesh;
		u = unknowns;
		uinf = uinit;
		rhsel = rhsel_in;
		fluxes.setup(m->gnaface(), u->cols(), ROWMAJOR);
	}

	void setup(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* uinit, Matrix<double>* rhsel_in, Matrix<double>* cfl_den)
	{
		nvars = unknowns->cols();
		m = mesh;
		u = unknowns;
		uinf = uinit;
		rhsel = rhsel_in;
		fluxes.setup(m->gnaface(), u->cols(), ROWMAJOR);
		cflden = cfl_den;
	}

	void compute_fluxes()
	{
		//cout << "VanLeerFlux: compute_fluxes(): Computing fluxes at boundary faces\n";
		//loop over boundary faces. Need to account for boundary conditions here
		double nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj;
		int lel, rel;
		Matrix<double> fiplus(u->cols(),1,ROWMAJOR);
		Matrix<double> fjminus(u->cols(),1,ROWMAJOR);
		Matrix<double> ug(1, u->cols(), ROWMAJOR);		// stores ghost state

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

// Van Leer fluxes with MUSCL reconstruction for 2nd order accuracy
class VanLeerFlux2
{
	UTriMesh* m;
	Matrix<double>* u;				// nelem x nvars matrix holding values of unknowns at each cell
	Matrix<double> fluxes;			// stores flux at each face
	Matrix<double>* rhsel;
	Matrix<double>* uinf;			// a state vector which has initial conditions
	Matrix<double> ug;				// holds ghost states for each boundary face
	Matrix<double>* cflden;			// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each face
	int nvars;

	// The 4 arrays below hold centroids of elements - first 2 for real elements and next 2 for ghost elements
	Matrix<double> xi;
	Matrix<double> yi;
	Matrix<double> xb;
	Matrix<double> yb;

	int ngauss;						// number of gauss points per face
	Matrix<double> gaussx;			// holds coordinates of gauss points
	Matrix<double> gaussy;

	Matrix<double>* dudx;
	Matrix<double>* dudy;
	Matrix<double>* w2x2;
	Matrix<double>* w2y2;
	Matrix<double>* w2xy;

	Matrix<double> ufl;				// left data at each face
	Matrix<double> ufr;				// right data at each face

	//Limiter stuff:
	Limiter lim;
	Matrix<double> phi_l;
	Matrix<double> phi_r;

public:
	VanLeerFlux2() {}

	// DO NOT USE this constructor - use setup() instead
	VanLeerFlux2(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* derx, Matrix<double>* dery, Matrix<double>* _w2x2, Matrix<double>* _w2y2, Matrix<double>* _w2xy, Matrix<double>* uinit, Matrix<double>* rhsel_in, Matrix<double>* cfl_den, int gauss_order)
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
		fluxes.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ufl.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ufr.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ug.setup(m->gnbface(),u->cols(),ROWMAJOR);
		cflden = cfl_den;
		ngauss = gauss_order;
		gaussx.setup(m->gnaface(),ngauss,ROWMAJOR);
		gaussx.setup(m->gnaface(),ngauss,ROWMAJOR);
		// The 4 arrays below hold centroids of elements - first 2 for real elements and next 2 for ghost elements
		xi.setup(m->gnelem(),1,ROWMAJOR);
		yi.setup(m->gnelem(),1,ROWMAJOR);
		xb.setup(m->gnbface(),1,ROWMAJOR);
		yb.setup(m->gnbface(),1,ROWMAJOR);
	}

	void setup(UTriMesh* mesh, Matrix<double>* unknowns, Matrix<double>* derx, Matrix<double>* dery, Matrix<double>* _w2x2, Matrix<double>* _w2y2, Matrix<double>* _w2xy, Matrix<double>* uinit, Matrix<double>* rhsel_in, Matrix<double>* cfl_den, int gauss_order)
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
		//fluxes.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ufl.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ufr.setup(m->gnaface(), u->cols(), ROWMAJOR);
		ug.setup(m->gnbface(),u->cols(),ROWMAJOR);
		cflden = cfl_den;
		ngauss = gauss_order;
		gaussx.setup(m->gnaface(),ngauss,ROWMAJOR);
		gaussy.setup(m->gnaface(),ngauss,ROWMAJOR);
		xi.setup(m->gnelem(),1,ROWMAJOR);
		yi.setup(m->gnelem(),1,ROWMAJOR);
		xb.setup(m->gnbface(),1,ROWMAJOR);
		yb.setup(m->gnbface(),1,ROWMAJOR);
		phi_l.setup(m->gnaface(), nvars, ROWMAJOR);
		phi_r.setup(m->gnaface(), nvars, ROWMAJOR);

		//lim.setup_limiter(m, u, &ug, &xb, &yb, &xi, &yi, &phi);

		// Calculate centroids of elements and coordinates of gauss points
		// Boundary faces
		//cout << "VanLeerFlux2: setup(): Calculating centroids around boundary faces\n";
		int ielem, ied; double nx, ny, x1,x2,y1,y2, xs,ys;
		for(ied = 0; ied < m->gnbface(); ied++)
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
			if(dabs(nx)>1e-8 && dabs(ny)>1e-8)  // check if nx = 0
			{
				xs = ( yi(ielem)-y1 - ny/nx*xi(ielem) + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
				ys = ny/nx*xs + yi(ielem) - ny/nx*xi(ielem);
			}
			else if(dabs(nx)<=1e-8)
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
		for(ielem = 0; ielem < m->gnelem(); ielem++)
		{
			/*int ielem = m->gintfac(ied,0);
			int jelem = m->gintfac(ied,1);*/
			xi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi(ielem) = xi(ielem) / 3.0;
			yi(ielem) = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi(ielem) = yi(ielem) / 3.0;

			/*xi(jelem) = m->gcoords(m->ginpoel(jelem, 0), 0) + m->gcoords(m->ginpoel(jelem, 1), 0) + m->gcoords(m->ginpoel(jelem, 2), 0);
			xi(jelem) /= 3.0;
			yi(jelem) = m->gcoords(m->ginpoel(jelem, 0), 1) + m->gcoords(m->ginpoel(jelem, 1), 1) + m->gcoords(m->ginpoel(jelem, 2), 1);
			yi(jelem) /= 3.0;*/
		}
		//xi.mprint(); yi.mprint();

		//cout << "VanLeerFlux2: setup(): Calculating gauss points\n";
		//Calculate and store coordinates of GAUSS POINTS (general implementation)
		//double x1,y1,x2,y2;
		int ig;
		for(ied = 0; ied < m->gnaface(); ied++)
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

	void convert_to_primitive(Matrix<double>* u, Matrix<double>* uprim)
	{
		int iel;
		for(iel = 0; iel < m->gnelem(); iel++)
		{
			(*uprim)(iel,0) = u->get(iel,0);	// density
			(*uprim)(iel,3) = (g-1)*(u->get(iel,3) - 0.5*(pow(u->get(iel,1),2)+pow(u->get(iel,2),2))/u->get(iel,0));	// pressure
			(*uprim)(iel,1) = u->get(iel,1)/u->get(iel,0);		// x-velocity
			(*uprim)(iel,2) = u->get(iel,2)/u->get(iel,0);		// y-velocity
		}
	}

	void convert_to_conserved(Matrix<double>* upr, Matrix<double>* u)
	{
		int iel;
		double vmag2;
		for(iel = 0; iel < m->gnelem(); iel++)
		{
			(*u)(iel,0) = upr->get(iel,0);		// density
			vmag2 = upr->get(iel,1)*upr->get(iel,1) + upr->get(iel,2)*upr->get(iel,2);		// square of magnitude of velocity
			(*u)(iel,1) = upr->get(iel,1)*upr->get(iel,0);		// x-momentum per unit volume
			(*u)(iel,2) = upr->get(iel,2)*upr->get(iel,0);		// y-momentum per unit volume
			(*u)(iel,3) = upr->get(iel,3)/(g-1.0) + 0.5*upr->get(iel,0)*vmag2;		// energy per unit volume
		}
	}

	void compute_ghostcell_states()		// for computing RHS of least-squares problem
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
				} */

				// Naive way
				for(int i = 0; i < nvars; i++)
					ug(ied,i) = uinf->get(0,i);
			}
		}
		//cout << "VanLeerFlux2: compute_ghostcell_states(): Done\n";
		//ug.mprint();
	}

	// Calculate gradients by Least-squares reconstruction
	void compute_ls_reconstruction(Matrix<double>* u)		// u should be a nelem x nvars matrix
	{
		//cout << "VanLeerFlux2: compute_ls_gradients(): Computing ghost cell data...\n";
		Matrix<double> w2yu(m->gnelem(),nvars,ROWMAJOR), w2xu(m->gnelem(),nvars,ROWMAJOR);
		w2xu.zeros(); w2yu.zeros();

		//calculate ghost states
		//compute_ghostcell_states();		// only needed for limiter now - CHECK

		// Boundary faces
		int ielem; double nx, ny;
		/*for(int ied = 0; ied < m->gnbface(); ied++)
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
		}*/

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
				(*dudx)(ielem,i) =  (w2xu(ielem,i) - (*w2xy)(ielem)*(*dudy)(ielem,i))/(*w2x2)(ielem);
			}

		//dudx->mprint(); dudy->mprint();
		//compute limiters
		lim.setup_limiter(m, u, &ug, dudx, dudy, &xb, &yb, &xi, &yi, &gaussx, &gaussy, &phi_l, &phi_r, &ufl, &ufr, uinf, g);
		//lim.compute_limiters();
		lim.compute_unlimited_interface_values();

		// cout << "VanLeerFlux2: compute_ls_gradients(): Computing values at faces - internal\n";
	}


	//TODO: generalize flux calculation to multiple gauss points
	void compute_fluxes()
	{
		//cout << "VanLeerFlux2: compute_fluxes(): Computing fluxes at boundary faces\n";

		//FIrst compute gradients and values at interfaces. This calculates ghost states as well (in ufr).
		compute_ls_reconstruction(u);

		//loop over boundary faces
		double nx, ny, len, pi, ci, vni, Mni, pj, cj, vnj, Mnj;
		int lel, rel;

		int ied;

		Matrix<double>* ufr = &(VanLeerFlux2::ufr);
		Matrix<double>* ufl = &(VanLeerFlux2::ufl);
		#ifdef _OPENMP
		Matrix<double>* rhsel = VanLeerFlux2::rhsel;
		Matrix<double>* cflden = VanLeerFlux2::cflden;
		UTriMesh* m = VanLeerFlux2::m;
		#endif

		#pragma omp parallel for default(none) private(ied,nx,ny,lel,rel,len,pi,ci,vni,Mni,pj,cj,vnj,Mnj) shared(m,ufl,ufr,rhsel,cflden) num_threads(NTHREADS)
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			Matrix<double> fiplus(u->cols(),1,ROWMAJOR);
			Matrix<double> fjminus(u->cols(),1,ROWMAJOR);
			Matrix<double> fluxes(nvars,1);

			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			//int rel = m->gintfac(ied,1);	// right element - does not exist

			pi = (g-1)*((*ufl)(ied,3) - 0.5*(pow((*ufl)(ied,1),2)+pow((*ufl)(ied,2),2))/(*ufl)(ied,0));
			ci = sqrt(g*pi/(*ufl)(ied,0));
			vni = ((*ufl)(ied,1)*nx + (*ufl)(ied,2)*ny)/(*ufl)(ied,0);
			Mni = vni/ci;

			// other ghost state quantities
			pj = (g-1)*((*ufr)(ied,3) - 0.5*(pow((*ufr)(ied,1),2)+pow((*ufr)(ied,2),2))/(*ufr)(ied,0));
			cj = sqrt(g*pj/(*ufr)(ied,0));
			vnj = ((*ufr)(ied,1)*nx + (*ufr)(ied,2)*ny)/(*ufr)(ied,0);
			Mnj = vnj/cj;

			//cout << " " << cj;

			//calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = (*ufl)(ied,0)*vni;
				fiplus(1) = vni*(*ufl)(ied,1) + pi*nx;
				fiplus(2) = vni*(*ufl)(ied,2) + pi*ny;
				fiplus(3) = vni*((*ufl)(ied,3) + pi);
			}
			else
			{
				double vmags = pow((*ufl)(ied,1)/(*ufl)(ied,0), 2) + pow((*ufl)(ied,2)/(*ufl)(ied,0), 2);	// square of velocity magnitude
				fiplus(0) = (*ufl)(ied,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * ((*ufl)(ied,1)/(*ufl)(ied,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * ((*ufl)(ied,2)/(*ufl)(ied,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = (*ufr)(ied,0)*vnj;
				fjminus(1) = vnj*(*ufr)(ied,1) + pj*nx;
				fjminus(2) = vnj*(*ufr)(ied,2) + pj*ny;
				fjminus(3) = vnj*((*ufr)(ied,3) + pj);
			}
			else
			{
				double vmags = pow((*ufr)(ied,1)/(*ufr)(ied,0), 2) + pow((*ufr)(ied,2)/(*ufr)(ied,0), 2);	// square of velocity magnitude
				fjminus(0) = -(*ufr)(ied,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * ((*ufr)(ied,1)/(*ufr)(ied,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * ((*ufr)(ied,2)/(*ufr)(ied,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < nvars; i++)
				fluxes(i) = (fiplus(i) + fjminus(i));

			//Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(i) *= len;

			// scatter the flux to elements' boundary integrals
			for(int i = 0; i < nvars; i++)
				#pragma omp atomic
				(*rhsel)(lel,i) -= fluxes(i);

			// calculate integ for CFL purposes
			//(*cflden)(lel,0) += (dabs(vni + vnj)/2.0 + (ci+cj)/2.0)*len;
			#pragma omp atomic
			(*cflden)(lel,0) += (dabs(vni) + ci)*len;
		}

		//cout << "VanLeerFlux2: compute_fluxes(): Computing fluxes at internal faces\n";
		//loop over internal faces
		#pragma omp parallel for default(none) private(ied,nx,ny,lel,rel,len,pi,ci,vni,Mni,pj,cj,vnj,Mnj) shared(m,ufl,ufr,rhsel,cflden) num_threads(NTHREADS)
		for(int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			nx = m->ggallfa(ied,0);
			ny = m->ggallfa(ied,1);
			len = m->ggallfa(ied,2);
			lel = m->gintfac(ied,0);	// left element
			rel = m->gintfac(ied,1);	// right element

			Matrix<double> fiplus(u->cols(),1,ROWMAJOR);
			Matrix<double> fjminus(u->cols(),1,ROWMAJOR);
			Matrix<double> fluxes(nvars,1);

			//calculate presures from u
			pi = (g-1)*((*ufl)(ied,3) - 0.5*(pow((*ufl)(ied,1),2)+pow((*ufl)(ied,2),2))/(*ufl)(ied,0));
			pj = (g-1)*((*ufr)(ied,3) - 0.5*(pow((*ufr)(ied,1),2)+pow((*ufr)(ied,2),2))/(*ufr)(ied,0));
			//calculate speeds of sound
			ci = sqrt(g*pi/(*ufl)(ied,0));
			cj = sqrt(g*pj/(*ufr)(ied,0));
			//calculate normal velocities
			vni = ((*ufl)(ied,1)*nx +(*ufl)(ied,2)*ny)/(*ufl)(ied,0);
			vnj = ((*ufr)(ied,1)*nx + (*ufr)(ied,2)*ny)/(*ufr)(ied,0);

			//Normal mach numbers
			Mni = vni/ci;
			Mnj = vnj/cj;

			//Calculate split fluxes
			if(Mni < -1.0) fiplus.zeros();
			else if(Mni > 1.0)
			{
				fiplus(0) = (*ufl)(ied,0)*vni;
				fiplus(1) = vni*(*ufl)(ied,1) + pi*nx;
				fiplus(2) = vni*(*ufl)(ied,2) + pi*ny;
				fiplus(3) = vni*((*ufl)(ied,3) + pi);
			}
			else
			{
				double vmags = pow((*ufl)(ied,1)/(*ufl)(ied,0), 2) + pow((*ufl)(ied,2)/(*ufl)(ied,0), 2);	// square of velocity magnitude
				fiplus(0) = (*ufl)(ied,0)*ci*pow(Mni+1, 2)/4.0;
				fiplus(1) = fiplus(0) * ((*ufl)(ied,1)/(*ufl)(ied,0) + nx*(2.0*ci - vni)/g);
				fiplus(2) = fiplus(0) * ((*ufl)(ied,2)/(*ufl)(ied,0) + ny*(2.0*ci - vni)/g);
				fiplus(3) = fiplus(0) * ( (vmags - vni*vni)/2.0 + pow((g-1)*vni+2*ci, 2)/(2*(g*g-1)) );
			}

			if(Mnj > 1.0) fjminus.zeros();
			else if(Mnj < -1.0)
			{
				fjminus(0) = (*ufr)(ied,0)*vnj;
				fjminus(1) = vnj*(*ufr)(ied,1) + pj*nx;
				fjminus(2) = vnj*(*ufr)(ied,2) + pj*ny;
				fjminus(3) = vnj*((*ufr)(ied,3) + pj);
			}
			else
			{
				double vmags = pow((*ufr)(ied,1)/(*ufr)(ied,0), 2) + pow((*ufr)(ied,2)/(*ufr)(ied,0), 2);	// square of velocity magnitude
				fjminus(0) = -(*ufr)(ied,0)*cj*pow(Mnj-1, 2)/4.0;
				fjminus(1) = fjminus(0) * ((*ufr)(ied,1)/(*ufr)(ied,0) + nx*(-2.0*cj - vnj)/g);
				fjminus(2) = fjminus(0) * ((*ufr)(ied,2)/(*ufr)(ied,0) + ny*(-2.0*cj - vnj)/g);
				fjminus(3) = fjminus(0) * ( (vmags - vnj*vnj)/2.0 + pow((g-1)*vnj-2*cj, 2)/(2*(g*g-1)) );
			}

			//Update the flux vector
			for(int i = 0; i < 4; i++)
				fluxes(i) = (fiplus(i) + fjminus(i));

			//TODO: Integrate the fluxes here using Quadrature2D class; not needed in case of FVM.
			for(int i = 0; i < nvars; i++)
				fluxes(i) *= len;

			// scatter the flux to elements' boundary integrands
			for(int i = 0; i < nvars; i++)
			{
				//rhsel->operator()(lel,i) -= fluxes(ied,i);
				#pragma omp atomic
				(*rhsel)(lel,i) -= fluxes(i);
				#pragma omp atomic
				(*rhsel)(rel,i) += fluxes(i);
			}

			// calculate integ for CFL purposes
			#pragma omp atomic
			(*cflden)(lel,0) += (dabs(vni) + ci)*len;
			#pragma omp atomic
			(*cflden)(rel,0) += (dabs(vnj) + cj)*len;
		}
		//cout << "VanLeerFlux2: compute_fluxes(): Fluxes computed.\n";
	}

	Matrix<double> centroid_x_coords() {
		return xi;
	}
	Matrix<double> centroid_y_coords() {
		return yi;
	}
};

} // end namespace acfd

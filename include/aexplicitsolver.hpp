/** @brief Implements a driver class for explicit solution of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
 */

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#ifndef __ANUMERICALFLUX_H
#include <anumericalflux.hpp>
#endif

#ifndef __ALIMITER_H
#include <alimiter.hpp>
#endif

#ifndef __ARECONSTRUCTION_H
#include <areconstruction.hpp>
#endif

#define __AEXPLICITSOLVER_H 1

namespace acfd {

/// A driver class to control the explicit time-stepping solution using TVD Runge-Kutta time integration
class ExplicitSolver
{
	UTriMesh* m;
	Matrix<double> m_inverse;		///< Left hand side (just the volume of the element for FV)
	Matrix<double> r_domain;		///< Domain integral in the RHS, zero for FV
	Matrix<double> r_boundary;		///< Boundary integral in the RHS, the only contributor to RHS in case of FV
	int nvars;						///< number of conserved variables
	Matrix<double> uinf;			///< Free-stream/reference condition

	/// stores \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face
	Matrix<double> integ;

	/// Flux (boundary integral) calculation context
	InviscidFlux* inviflux;

	/// Reconstruction context
	Reconstruction* rec;

	/// Cell centers
	Matrix<double> xc;
	/// Cell centers
	Matrix<double> yc;

	/// Ghost cell centers
	Matrix<double> xcg;
	/// Ghost cell centers
	Matrix<double> ycg;
	/// Ghost cell flow quantities
	Matrix<double> ug;

	/// Number of Guass points per face
	int ngaussf;
	/// Faces' Gauss points - x coords
	Matrix<double> gx;
	/// Faces' Gauss points - y coords
	Matrix<double> gy;

	/// Left state at each face (assuming 1 Gauss point per face)
	Matrix<double> uleft;
	/// Rigt state at each face (assuming 1 Gauss point per face)
	Matrix<double> uright;
	
	/// vector of unknowns
	Matrix<double> u;
	/// x-slopes
	Matrix<double> dudx;
	/// y-slopes
	Matrix<double> dudy;

public:
	ExplicitSolver();
	~ExplicitSolver();
	void loaddata(double Minf, double vinf, double a, double rhoinf);
	void compute_ghostcell_states();
};

ExplicitSolver::ExplicitSolver(UTriMesh* mesh)
{
	m = mesh;
	cout << "EulerFV: Computing required mesh data...\n";
	m->compute_jacobians();
	m->compute_face_data();
	cout << "EulerFV: Mesh data computed.\n";

	// for 2D Euler equations, we have 4 variables
	nvars = 4;
	// for upto second-order finite volume, we only need 1 Guass point per face
	ngaussf = 1;

	m_inverse.setup(m->gnelem(),1,ROWMAJOR);		// just a vector for FVM. For DG, this will be an array of Matrices
	r_domain.setup(m->gnelem(),nvars,ROWMAJOR);
	r_boundary.setup(m->gnelem(),nvars,ROWMAJOR);
	u.setup(m->gnelem(), nvars, ROWMAJOR);
	uinf.setup(1, nvars, ROWMAJOR);
	integ.setup(m->gnelem(), 1,ROWMAJOR);
	dudx.setup(m->gnelem(), nvars, ROWMAJOR);
	dudy.setup(m->gnelem(), nvars, ROWMAJOR);

	xc.setup(m->gnelem(),1);
	yc.setup(m->gnelem(),1);
	xcg.setup(m->gnface(),1);
	ycg.setup(m->gnface(),1);
	ug.setup(m->gnface(),nvars);
	gx.setup(m->gnaface(), ngaussf);
	gy.setup(m->gnaface(), ngaussf);

	for(int i = 0; i < m->gnelem(); i++)
		m_inverse(i) = 2.0/mesh->jacobians(i);

	inviflux = new VanLeerFlux();
	inviflux->setup(m, &u, &dudx, &dudy, &w2x2, &w2y2, &w2xy, &uinf, &r_boundary, &integ, ngaussf);	// 1 gauss point

	rec = new GreenGaussReconstruction();
	rec->setup(m, &u, &ug, &dudx, &dudy, &xc, &yc, &xcg, &ycg);
}

ExplicitSolver::~ExplicitSolver()
{
	delete rec;
	delete inviflux;
}

/// Function to feed needed data, and compute cell-centers
/** \param Minf Free-stream Mach number
 * \param vinf Free stream velocity magnitude
 * \param a Angle of attack (radians)
 * \param rhoinf Free stream density
 */
void ExplicitSolver::loaddata(double Minf, double vinf, double a, double rhoinf)
{
	// Note that reference density and reference velocity are the values at infinity
	//cout << "EulerFV: loaddata(): Calculating initial data...\n";
	double vx = vinf*cos(a);
	double vy = vinf*sin(a);
	double p = rhoinf*vinf*vinf/(g*Minf*Minf);
	uinf(0,0) = rhoinf;		// should be 1
	uinf(0,1) = rhoinf*vx;
	uinf(0,2) = rhoinf*vy;
	uinf(0,3) = p/(g-1) + 0.5*rhoinf*vinf*vinf;

	//initial values are equal to boundary values
	for(int i = 0; i < m->gnelem(); i++)
		for(int j = 0; j < nvars; j++)
			u(i,j) = uinf(0,j);

	// Next, get cell centers (real and ghost)
	
	for(int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		xc(ielem) = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
		xc(ielem) = xc(ielem) / 3.0;
		yc(ielem) = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
		yc(ielem) = yc(ielem) / 3.0;
	}

	int ied, ig, ielem;
	double x1, y1, x2, y2, xs, ys, xi, yi;

	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0); //int lel = ielem;
		//jelem = m->gintfac(ied,1); //int rel = jelem;
		double nx = m->ggallfa(ied,0);
		double ny = m->ggallfa(ied,1);

		xi = xc.get(ielem);
		yi = yc.get(ielem);

		// Note: The ghost cell is a direct reflection of the boundary cell about the boundary-face
		//       It is NOT the reflection about the midpoint of the boundary-face
		x1 = m->gcoords(m->gintfac(ied,2),0);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		y2 = m->gcoords(m->gintfac(ied,3),1);

		if(dabs(nx)>A_SMALL_NUMBER && dabs(ny)>A_SMALL_NUMBER)		// check if nx != 0 and ny != 0
		{
			xs = ( yi-y1 - ny/nx*xi + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
			ys = ny/nx*xs + yi - ny/nx*xi;
		}
		else if(dabs(nx)<=A_SMALL_NUMBER)
		{
			xs = xi;
			ys = y1;
		}
		else
		{
			xs = x1;
			ys = yi;
		}
		xcg(ied) = 2*xs-xi;
		ycg(ied) = 2*ys-yi;
	}
	
	//Calculate and store coordinates of Gauss points (general implementation)
	// Gauss points are uniformly distributed along the face.
	for(ied = 0; ied < m->gnaface(); ied++)
	{
		x1 = m->gcoords(m->gintfac(ied,2),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y2 = m->gcoords(m->gintfac(ied,3),1);
		for(ig = 0; ig < ngaussf; ig++)
		{
			gx(ied,ig) = x1 + (double)(ig+1)/(double)(ngaussf+1) * (x2-x1);
			gy(ied,ig) = y1 + (double)(ig+1)/(double)(ngaussf+1) * (y2-y1);
		}
	}

	cout << "ExplicitSolver: loaddata(): Initial data calculated.\n";
}

/// Computes flow variables in ghost cells using characteristic boundary conditions
void ExplicitSolver::compute_ghostcell_states()
{
	//cout << "EulerFV: compute_ghostcell_states()\n";
	int lel, rel;
	double nx, ny, vni, pi, ci, Mni, vnj, pj, cj, Mnj, vinfx, vinfy, vinfn, vbn, pinf, pb, cinf, cb, vgx, vgy, vbx, vby;
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		lel = m->gintfac(ied,0);
		nx = m->ggallfa(ied,0);
		ny = m->ggallfa(ied,1);

		vni = (u.get(lel,1)*nx + u.get(lel,2)*ny)/u.get(lel,0);
		pi = (g-1)*(u.get(lel,3) - 0.5*(pow(u.get(lel,1),2)+pow(u.get(lel,2),2))/u.get(lel,0));
		ci = sqrt(g*pi/u.get(lel,0));
		Mni = vni/ci;

		if(m->ggallfa(ied,3) == 2)		// solid wall
		{
			ug(ied,0) = u.get(lel,0);
			ug(ied,1) = u.get(lel,1) - 2*vni*nx*ug(ied,0);
			ug(ied,2) = u.get(lel,2) - 2*vni*ny*ug(ied,0);
			ug(ied,3) = u.get(lel,3);
		}

		if(m->ggallfa(ied,3) == 4)		// inflow or outflow
		{
			/*if(Mni < -1.0)
			{
				for(int i = 0; i < nvars; i++)
					ug(0,i) = uinf.get(0,i);
				pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
				cj = sqrt(g*pj/ug(0,0));
				vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
			}
			else if(Mni >= -1.0 && Mni < 0.0)
			{
				vinfx = uinf.get(0,1)/uinf.get(0,0);
				vinfy = uinf.get(0,2)/uinf.get(0,0);
				vinfn = vinfx*nx + vinfy*ny;
				vbn = u.get(lel,1)/u.get(lel,0)*nx + u.get(lel,2)/u.get(lel,0)*ny;
				pinf = (g-1)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
				pb = (g-1)*(u.get(lel,3) - 0.5*(pow(u.get(lel,1),2)+pow(u.get(lel,2),2))/u.get(lel,0));
				cinf = sqrt(g*pinf/uinf.get(0,0));
				cb = sqrt(g*pb/u.get(lel,0));

				vgx = vinfx*ny*ny - vinfy*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
				vgy = vinfy*nx*nx - vinfx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
				vnj = vgx*nx + vgy*ny;	// = vgn
				cj = (g-1)/2*(vnj-vinfn)+cinf;
				ug(0,0) = pow( pinf/pow(uinf.get(0,0),g) * 1.0/cj*cj , 1/(1-g));	// density
				pj = ug.get(0,0)/g*cj*cj;

				ug(0,3) = pj/(g-1) + 0.5*ug.get(0,0)*(vgx*vgx+vgy*vgy);
				ug(0,1) = ug.get(0,0)*vgx;
				ug(0,2) = ug.get(0,0)*vgy;
			}
			else if(Mni >= 0.0 && Mni < 1.0)
			{
				vbx = u.get(lel,1)/u.get(lel,0);
				vby = u.get(lel,2)/u.get(lel,0);
				vbn = vbx*nx + vby*ny;
				vinfn = uinf.get(0,1)/uinf.get(0,0)*nx + uinf.get(0,2)/uinf.get(0,0)*ny;
				pinf = (g-1)*(uinf.get(0,3) - 0.5*(pow(uinf.get(0,1),2)+pow(uinf.get(0,2),2))/uinf.get(0,0));
				pb = (g-1)*(u.get(lel,3) - 0.5*(pow(u.get(lel,1),2)+pow(u.get(lel,2),2))/u.get(lel,0));
				cinf = sqrt(g*pinf/uinf.get(0,0));
				cb = sqrt(g*pb/u.get(lel,0));

				vgx = vbx*ny*ny - vby*nx*ny + (vbn+vinfn)/2.0*nx + (cb - cinf)/(g-1)*nx;
				vgy = vby*nx*nx - vbx*nx*ny + (vbn+vinfn)/2.0*ny + (cb - cinf)/(g-1)*ny;
				vnj = vgx*nx + vgy*ny;	// = vgn
				cj = (g-1)/2*(vnj-vinfn)+cinf;
				ug(0,0) = pow( pb/pow(u.get(lel,0),g) * 1.0/cj*cj , 1/(1-g));	// density
				pj = ug(0,0)/g*cj*cj;

				ug(0,3) = pj/(g-1) + 0.5*ug(0,0)*(vgx*vgx+vgy*vgy);
				ug(0,1) = ug(0,0)*vgx;
				ug(0,2) = ug(0,0)*vgy;
			}
			else
			{
				for(int i = 0; i < nvars; i++)
					ug(0,i) = u.get(lel,i);
				pj = (g-1)*(ug(0,3) - 0.5*(pow(ug(0,1),2)+pow(ug(0,2),2))/ug(0,0));
				cj = sqrt(g*pj/ug(0,0));
				vnj = (ug(0,1)*nx + ug(0,2)*ny)/ug(0,0);
			} */

			// Naive way
			for(int i = 0; i < nvars; i++)
				ug(ied,i) = uinf.get(0,i);
		}
	}
}
}

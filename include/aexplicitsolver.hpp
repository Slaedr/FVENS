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
/** \note Make sure compute_topological(), compute_face_data() and compute_jacobians() have been called on the mesh object prior to initialzing an object of this class.
 */
class ExplicitSolver
{
	const UTriMesh* m;
	amat::Matrix<acfd_real> m_inverse;		///< Left hand side (just the volume of the element for FV)
	amat::Matrix<acfd_real> residual;			///< Right hand side for boundary integrals and source terms
	int nvars;							///< number of conserved variables
	amat::Matrix<acfd_real> uinf;				///< Free-stream/reference condition

	/// stores \f$ \sum_{j \in \partial\Omega_I} \int_j( |v_n| + c) d \Gamma \f$, where v_n and c are average values for each face
	amat::Matrix<acfd_real> integ;

	/// Flux (boundary integral) calculation context
	InviscidFlux* inviflux;

	/// Reconstruction context
	Reconstruction* rec;

	/// Limiter context
	FaceDataComputation* lim;

	/// Cell centers
	amat::Matrix<acfd_real> xc;
	/// Cell centers
	amat::Matrix<acfd_real> yc;

	/// Ghost cell centers
	amat::Matrix<acfd_real> xcg;
	/// Ghost cell centers
	amat::Matrix<acfd_real> ycg;
	/// Ghost cell flow quantities
	amat::Matrix<acfd_real> ug;

	/// Number of Guass points per face
	int ngaussf;
	/// Faces' Gauss points - x coords
	amat::Matrix<acfd_real> gx;
	/// Faces' Gauss points - y coords
	amat::Matrix<acfd_real> gy;

	/// Left state at each face (assuming 1 Gauss point per face)
	amat::Matrix<acfd_real> uleft;
	/// Rigt state at each face (assuming 1 Gauss point per face)
	amat::Matrix<acfd_real> uright;
	
	/// vector of unknowns
	amat::Matrix<acfd_real> u;
	/// x-slopes
	amat::Matrix<acfd_real> dudx;
	/// y-slopes
	amat::Matrix<acfd_real> dudy;

	int order;					///< Formal order of accuracy of the scheme (1 or 2)

	int solid_wall_id;			///< Boundary marker corresponding to solid wall
	int inflow_outflow_id;		///< Boundary marker corresponding to inflow/outflow

public:
	ExplicitSolver();
	~ExplicitSolver();
	void loaddata(acfd_real Minf, acfd_real vinf, acfd_real a, acfd_real rhoinf);

	/// Computes flow variables at boundaries (either Gauss points or ghost cell centers) using the inteior state provided
	/** \param[in] instates provides the left (interior state) for each boundary face
	 * \param[out] bounstates will contain the right state of boundary faces
	 *
	 * Currently does not use characteristic BCs.
	 * \todo Implement and test characteristic BCs
	 */
	void compute_boundary_states(const amat::Matrix<acfd_real>& instates, amat::Matrix<acfd_real>& bounstates);

	/// Calls functions to assemble the [right hand side](@ref residual)
	void compute_RHS();
	
	/// Computes the left and right states at each face, using the [reconstruction](@ref rec) and [limiter](@ref limiter) objects
	void ExplicitSolver::compute_face_states()

	/// Solves a steady problem by an explicit method first order in time, using local time-stepping
	void solve_rk1_steady(const acfd_real tol, const acfd_real cfl);

	acfd_real l2norm(const amat::Matrix<acfd_real>* const v);
};

ExplicitSolver::ExplicitSolver(const UTriMesh* mesh, const int _order)
{
	m = mesh;
	order = _order;

	std::cout << "ExplicitSolver: Setting up explicit solver for spatial order " << order << std::endl;

	// for 2D Euler equations, we have 4 variables
	nvars = 4;
	// for upto second-order finite volume, we only need 1 Guass point per face
	ngaussf = 1;

	solid_wall_id = 2;
	inflow_outflow_id = 4;

	m_inverse.setup(m->gnelem(),1);		// just a vector for FVM. For DG, this will be an array of Matrices
	residual.setup(m->gnelem(),nvars);
	u.setup(m->gnelem(), nvars);
	uinf.setup(1, nvars);
	integ.setup(m->gnelem(), 1);
	dudx.setup(m->gnelem(), nvars);
	dudy.setup(m->gnelem(), nvars);
	uleft.setup(m->gnaface(), nvars);
	uright.setup(m->gnaface(), nvars);

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
	inviflux->setup(m, &u, &dudx, &dudy, &w2x2, &w2y2, &w2xy, &uinf, &residual, &integ, ngaussf);	// 1 gauss point

	rec = new GreenGaussReconstruction();
	rec->setup(m, &u, &ug, &dudx, &dudy, &xc, &yc, &xcg, &ycg);

	lim = new NoLimiter();
	lim->setup(m, &u, &ug, &dudx, &dudy, &xcg, &ycg, &xc, &yc, &gx, &gy, &uleft, &uright);
}

ExplicitSolver::~ExplicitSolver()
{
	delete rec;
	delete inviflux;
	delete lim;
}

/// Function to feed needed data, and compute cell-centers
/** \param Minf Free-stream Mach number
 * \param vinf Free stream velocity magnitude
 * \param a Angle of attack (radians)
 * \param rhoinf Free stream density
 */
void ExplicitSolver::loaddata(acfd_real Minf, acfd_real vinf, acfd_real a, acfd_real rhoinf)
{
	// Note that reference density and reference velocity are the values at infinity
	//cout << "EulerFV: loaddata(): Calculating initial data...\n";
	acfd_real vx = vinf*cos(a);
	acfd_real vy = vinf*sin(a);
	acfd_real p = rhoinf*vinf*vinf/(g*Minf*Minf);
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
	acfd_real x1, y1, x2, y2, xs, ys, xi, yi;

	for(ied = 0; ied < m->gnbface(); ied++)
	{
		ielem = m->gintfac(ied,0); //int lel = ielem;
		//jelem = m->gintfac(ied,1); //int rel = jelem;
		acfd_real nx = m->ggallfa(ied,0);
		acfd_real ny = m->ggallfa(ied,1);

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
			gx(ied,ig) = x1 + (acfd_real)(ig+1)/(acfd_real)(ngaussf+1) * (x2-x1);
			gy(ied,ig) = y1 + (acfd_real)(ig+1)/(acfd_real)(ngaussf+1) * (y2-y1);
		}
	}

	cout << "ExplicitSolver: loaddata(): Initial data calculated.\n";
}

void ExplicitSolver::compute_boundary_states(const amat::Matrix<acfd_real>& ins, amat::Matrix<acfd_real>& bs)
{
	int lel, rel;
	acfd_real nx, ny, vni, pi, ci, Mni, vnj, pj, cj, Mnj, vinfx, vinfy, vinfn, vbn, pinf, pb, cinf, cb, vgx, vgy, vbx, vby;
	for(int ied = 0; ied < m->gnbface(); ied++)
	{
		lel = m->gintfac(ied,0);
		nx = m->ggallfa(ied,0);
		ny = m->ggallfa(ied,1);

		vni = (ins.get(lel,1)*nx + ins.get(lel,2)*ny)/ins.get(lel,0);
		pi = (g-1)*(ins.get(lel,3) - 0.5*(pow(ins.get(lel,1),2)+pow(ins.get(lel,2),2))/ins.get(lel,0));
		ci = sqrt(g*pi/ins.get(lel,0));
		Mni = vni/ci;

		if(m->ggallfa(ied,3) == solid_wall_id)
		{
			bs(ied,0) = ins.get(lel,0);
			bs(ied,1) = ins.get(lel,1) - 2*vni*nx*bs(ied,0);
			bs(ied,2) = ins.get(lel,2) - 2*vni*ny*bs(ied,0);
			bs(ied,3) = ins.get(lel,3);
		}

		if(m->ggallfa(ied,3) == inflow_outflow_id)
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
				bs(ied,i) = uinf.get(0,i);
		}
	}
}

acfd_real ExplicitSolver::l2norm(const amat::Matrix<amat_real>* const v)
{
	acfd_real norm = 0;
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		norm += v->get(iel)*v->get(iel)*m->jacobians(iel)/2.0;
	}
	norm = sqrt(norm);
	return norm;
}

void ExplicitSolver::compute_RHS()
{
	residual.zeros();
	
	if(order == 2)
	{
		rec->compute_gradients();
		lim->compute_face_values();
	}
	else
	{
		// if order is 1, set the face data same as cell-centred data
		
		int ied, ielem, jelem, ivar;
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			for(ivar = 0; ivar < nvars; ivar++)
				uleft(ied,ivar) = u.get(ielem,ivar);
		}
		for(ied = m->gnbface(), ied < m->gnaface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			jelem = m->gintfac(ied,1);
			for(ivar = 0; ivar < nvars; ivar++)
			{
				uleft(ied,ivar) = u.get(ielem,ivar);
				uright(ied,ivar) = u.get(jelem,ivar);
			}
		}
	}

	compute_boundary_states(uleft,uright);

	flux->compute_fluxes();
}

void ExplicitSolver::compute_face_states()
{
	int iface, ivar;
}

void ExplicitSolver::solve_rk1_steady(const acfd_real tol, const acfd_real cfl)
{
	int step = 0;
	double resi = 1.0;
	double initres = 1.0;
	amat::Matrix<amat_real>* err;
	err = new amat::Matrix<amat_real>[nvars];
	for(int i = 0; i<nvars; i++)
		err[i].setup(m->gnelem(),1);
	amat::Matrix<amat_real> res(nvars,1);
	res.ones();
	amat::Matrix<amat_real> dtm(m->gnelem(), 1);		// for local time-stepping
	amat::Matrix<amat_real> uold(u.rows(), u.cols());

	while(resi/initres > tol)// || step <= 10)
	{
		//cout << "EulerFV: solve_rk1_steady(): Entered loop. Step " << step << endl;
		// reset fluxes
		integ.zeros();		// reset CFL data

		//calculate fluxes
		compute_RHS();		// this invokes Flux calculating function after zeroing the residuals

		/*if(step==0) {
			//dudx.mprint(); dudy.mprint();
			break;
		}*/

		//calculate dt based on CFL

		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
		}

		uold = u;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
			{
				u(iel,i) += dtm(iel)*m_inverse.get(iel)*residual.get(iel,i);
			}
		}

		//if(step == 0) { dudx.mprint();  break; }

		for(int i = 0; i < nvars; i++)
		{
			err[i] = (u-uold).col(i);
			res(i) = l2norm(&err[i]);
		}
		resi = res.max();

		if(step == 0)
			initres = resi;

		if(step % 10 == 0)
			std::cout << "EulerFV: solve_rk1_steady(): Step " << step << ", rel residual " << resi/initres << std::endl;

		step++;
		/*double totalenergy = 0;
		for(int i = 0; i < m->gnelem(); i++)
			totalenergy += u(i,3)*m->jacobians(i);
		cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
		//if(step == 10000) break;
	}

	//calculate gradients
	rec->compute_gradients();
	delete [] err;
}

}	// end namespace

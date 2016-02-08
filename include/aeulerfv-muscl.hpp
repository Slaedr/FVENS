/** @file aeulerfv-muscl.hpp
 * @brief Implements the main time loop for explicit time-integration of Euler equations to steady state.
 * @author Aditya Kashi
 * \note NOTE: Need to implement characteristic inflow/outflow BCs
 */

#ifndef _GLIBCXX_IOSTREAM
#include <iostream>
#endif

#ifndef _GLIBCXX_CMATH
#include <cmath>
#endif

#ifndef _GLIBCXX_STRING
#include <string>
#endif

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#include "anumericalflux.hpp"
//#include "atimeint.hpp"

using namespace std;
using namespace amat;
using namespace acfd;

namespace acfd {

/// set the flux calculation method here
typedef VanLeerFlux2 Flux;

//set time explicit stepping scheme here
//typedef TimeStepRK1 TimeStep;

/// Class encapsulating the main time-stepping loop for 2nd order solution of Euler equations
class EulerFV
{
	UTriMesh* m;
	Matrix<double> m_inverse;
	Matrix<double> r_domain;
	Matrix<double> r_boundary;
	int nvars;
	Matrix<double> uinf;
	/// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each face
	Matrix<double> integ;

	/// Flux (boundary integral) calculation context
	Flux flux;

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

	//Quantities needed for linear least-squares reconstruction.
	Matrix<double> w2xy;
	Matrix<double> w2x2;
	Matrix<double> w2y2;

public:
	/// vector of unknowns
	Matrix<double> u;
	/// x-slopes
	Matrix<double> dudx;
	/// y-slopes
	Matrix<double> dudy;
	/// postprocessing array - col 1 contains density, col 2 contains mach number, col 3 contains pressure.
	Matrix<double> scalars;
	/// postprocessing array for velocities
	Matrix<double> velocities;

	/// Setup function.
	/// \note Make sure jacobians and face data are computed!
	EulerFV(UTriMesh* mesh)
	{
		m = mesh;
		cout << "EulerFV: Computing required mesh data...\n";
		m->compute_jacobians();
		m->compute_face_data();
		cout << "EulerFV: Mesh data computed.\n";

		// for 2D Euler equations, we have 4 variables
		nvars = 4;

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
		xcg.setup(m->gbface(),1);
		ycg.setup(m->gbface(),1);
		ug.setup(m->gbface(),nvars);

		for(int i = 0; i < m->gnelem(); i++)
			m_inverse(i) = 2.0/mesh->jacobians(i);

		w2xy.setup(m->gnelem(),1,ROWMAJOR);
		w2x2.setup(m->gnelem(),1,ROWMAJOR);
		w2y2.setup(m->gnelem(),1,ROWMAJOR);
		w2x2.zeros();
		w2y2.zeros();
		w2xy.zeros();

		flux.setup(m, &u, &dudx, &dudy, &w2x2, &w2y2, &w2xy, &uinf, &r_boundary, &integ, 1);	// 1 gauss point
	}

	/// Function to feed needed data, and compute cell-centers
	/** \param Minf Free-stream Mach number
	 * \param vinf Free stream velocity magnitude
	 * \param a Angle of attack (radians)
	 * \param rhoinf Free stream density
	 */
	void loaddata(double Minf, double vinf, double a, double rhoinf)
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

		for(int ielem = 0; ielem < m->gnelem(); ielem++)
		{
			xc(ielem) = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xc(ielem) = xi(ielem) / 3.0;
			yc(ielem) = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yc(ielem) = yi(ielem) / 3.0;
		}

		double x1, y1, x2, y2, xs, ys;

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

		//cout << "EulerFV: loaddata(): Initial data calculated.\n";
	}

	void calculate_leastsquaresLHS()
	{
		cout << "EulerFV: calculate_leastsquaresLHS(): Calculating LHS terms for least-squares linear reconstruction...\n";
		double xi, yi, xj, yj;
		int jnode, ied, ielem, jelem;
		/*for(int ielem = 0; ielem < m->gnelem(); ielem++)
		{
			xi = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi = xi / 3.0;
			yi = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi = yi / 3.0;

			for(jnode = 0; jnode < m->gnnode(); jnode++)
			{
				jelem = m->gesuel(ielem,jnode);
				double xj, yj;

				// check whether the element is real or ghost
				if(jelem < m->gnelem())
				{
					xj = m->gcoords(m->ginpoel(jelem, 0), 0) + m->gcoords(m->ginpoel(jelem, 1), 0) + m->gcoords(m->ginpoel(jelem, 2), 0);
					xj = xj / 3.0;
					yj = m->gcoords(m->ginpoel(jelem, 0), 1) + m->gcoords(m->ginpoel(jelem, 1), 1) + m->gcoords(m->ginpoel(jelem, 2), 1);
					yj = yj / 3.0;
				}
				else
				{
					// Ghost cell is reflection of boundary cell about midpoint of boundary face
					// first get coords of endpoints of boundary face. We know jnode is the interior node

					double x1 = m->gcoords(m->ginpoel(ielem, perm(0,m->gnnode(),jnode,1)), 0);
					double y1 = m->gcoords(m->ginpoel(ielem, perm(0,m->gnnode(),jnode,1)), 1);
					double x2 = m->gcoords(m->ginpoel(ielem, perm(0,m->gnnode(),jnode,2)), 0);
					double y2 = m->gcoords(m->ginpoel(ielem, perm(0,m->gnnode(),jnode,2)), 1);
					//double xs = (y2-y1)/(x2-x2)*x1 +
					xj = x1+x2-xi;
					yj = y1+y2-yi;
				}
				// weight = ||Xj - Xi||^(-p), p = 1
				w2x2(ielem) += 1.0/(( (xj-xi)*(xj-xi)-(yj-yi)*(yj-yi) ) )*(xj-xi)*(xj-xi);
				w2y2(ielem) += 1.0/(( (xj-xi)*(xj-xi)-(yj-yi)*(yj-yi) ) )*(yj-yi)*(yj-yi);
				w2xy(ielem) += 1.0/(( (xj-xi)*(xj-xi)-(yj-yi)*(yj-yi) ) )*(xj-xi)*(yj-yi);
			}
		}*/

		// Boundary faces
		for(ied = 0; ied < m->gnbface(); ied++)
		{
			ielem = m->gintfac(ied,0); //int lel = ielem;
			//jelem = m->gintfac(ied,1); //int rel = jelem;
			double nx = m->ggallfa(ied,0);
			double ny = m->ggallfa(ied,1);

			xi = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi = xi / 3.0;
			yi = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi = yi / 3.0;

			// Note: The ghost cell is a direct reflection of the boundary cell about the boundary-face
			//       It is NOT the reflection about the midpoint of the boundary-face
			double x1 = m->gcoords(m->gintfac(ied,2),0);
			double x2 = m->gcoords(m->gintfac(ied,3),0);
			double y1 = m->gcoords(m->gintfac(ied,2),1);
			double y2 = m->gcoords(m->gintfac(ied,3),1);
			double xs, ys;
			if(dabs(nx)>A_SMALL_NUMBER && dabs(ny)>A_SMALL_NUMBER)
			{		// check if nx != 0
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
			xj = 2*xs-xi;
			yj = 2*ys-yi;

			w2x2(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xj-xi)*(xj-xi);
			w2y2(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(yj-yi)*(yj-yi);
			w2xy(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xj-xi)*(yj-yi);
		}

		// Internal faces
		for(ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			ielem = m->gintfac(ied,0);
			jelem = m->gintfac(ied,1);
			xi = m->gcoords(m->ginpoel(ielem, 0), 0) + m->gcoords(m->ginpoel(ielem, 1), 0) + m->gcoords(m->ginpoel(ielem, 2), 0);
			xi = xi / 3.0;
			yi = m->gcoords(m->ginpoel(ielem, 0), 1) + m->gcoords(m->ginpoel(ielem, 1), 1) + m->gcoords(m->ginpoel(ielem, 2), 1);
			yi = yi / 3.0;

			xj = m->gcoords(m->ginpoel(jelem, 0), 0) + m->gcoords(m->ginpoel(jelem, 1), 0) + m->gcoords(m->ginpoel(jelem, 2), 0);
			xj = xj / 3.0;
			yj = m->gcoords(m->ginpoel(jelem, 0), 1) + m->gcoords(m->ginpoel(jelem, 1), 1) + m->gcoords(m->ginpoel(jelem, 2), 1);
			yj = yj / 3.0;

			for(int i = 0; i < nvars; i++)
			{
				w2x2(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xj-xi)*(xj-xi);
				w2y2(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(yj-yi)*(yj-yi);
				w2xy(ielem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xj-xi)*(yj-yi);

				w2x2(jelem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xi-xj)*(xi-xj);
				w2y2(jelem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(yi-yj)*(yi-yj);
				w2xy(jelem) += 1.0/( (xj-xi)*(xj-xi)+(yj-yi)*(yj-yi) )*(xi-xj)*(yi-yj);
			}
		}
		/*w2x2.mprint();
		w2y2.mprint();
		w2xy.mprint();*/
		cout << "EulerFV: calculate_leastsquaresLHS(): Done.\n";
	}

	void calculate_r_domain()
	{
		// only needed for DG, not FV
		r_domain.zeros();
	}

	void calculate_r_boundary()
	{
		/*Matrix<double> rhsel(m->gnelem(), nvars, ROWMAJOR);		// stores boundary flux of each element
		rhsel.zeros(); */

		//flux.setup(m, &u, &uinf, &r_boundary, &integ);
		flux.compute_fluxes();		// Now r_boundary stores the required RHS integrals and integ stores integrals needed for dt computation

		//r_boundary = rhsel;
	}

	double l2norm(Matrix<double>* v)
	{
		double norm = 0;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			norm += v->get(iel)*v->get(iel)*m->jacobians(iel)/2.0;
		}
		norm = sqrt(norm);
		return norm;
	}


	/// Implements the main time-stepping loop
	/** Simple forward Euler (or 1-stage Runge-Kutta) time stepping.
	 * \param time Physical time upto which simulation should proceed
	 */
	void solve_rk1(double time, double cfl)
	{
		double telapsed = 0.0;
		int step = 0;
		calculate_leastsquaresLHS();

		while(telapsed <= time)
		{
			cout << "EulerFV: solve_rk1(): Step " << step << ", time elapsed " << telapsed << endl;

			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			integ.zeros();		// reset CFL data

			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();		// this invokes Flux calculating function

			//calculate dt based on CFL
			// Vector to hold time steps
			Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);

			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			// Finally get global time step
			double dt = dtm.min();
			cout << "EulerFV: solve_rk1(): Current time step = " << dt << endl;

			Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
			Matrix<double> R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) += dt*m_inverse.get(iel)*R.get(iel,i);
				}
			}

			telapsed += dt;
			step++;
			// if (step == 5)
			// 	break;
			/*double totalenergy = 0;
			for(int i = 0; i < m->gnelem(); i++)
				totalenergy += u(i,3)*m->jacobians(i);
			cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
		}

		//calculate gradients
		//flux.compute_ls_reconstruction(&u);
	}

	/// Time-stepping loop for steady-state
	void solve_rk1_steady(double tol, double cfl)
	{
		//double telapsed = 0.0;
		int step = 0;
		double resi = 1.0;
		double initres = 1.0;
		Matrix<double>* err;
		err = new Matrix<double>[nvars];
		for(int i = 0; i<nvars; i++)
			err[i].setup(m->gnelem(),1);
		Matrix<double> res(nvars,1);
		res.ones();
		Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);
		Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
		Matrix<double> R = r_domain + r_boundary;

		calculate_leastsquaresLHS();

		while(resi/initres > tol)// || step <= 10)
		{
			//cout << "EulerFV: solve_rk1_steady(): Entered loop. Step " << step << endl;
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			integ.zeros();		// reset CFL data

			//calculate fluxes
			//calculate_r_domain();		// not needed in FVM
			calculate_r_boundary();		// this invokes Flux calculating function

			/*if(step==0) {
				dudx.mprint(); dudy.mprint(); break;
			}*/

			//calculate dt based on CFL
			// Vector to hold time steps


			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) += dtm(iel)*m_inverse.get(iel)*R.get(iel,i);
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
				cout << "EulerFV: solve_rk1_steady(): Step " << step << ", rel residual " << resi/initres << endl;

			step++;
			/*double totalenergy = 0;
			for(int i = 0; i < m->gnelem(); i++)
				totalenergy += u(i,3)*m->jacobians(i);
			cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
			//if(step == 10000) break;
		}

		//calculate gradients
		flux.compute_ls_reconstruction();
		delete [] err;
	}

	void solve_rk3_steady(double tol, double cfl)
	{
		//double telapsed = 0.0;
		int step = 0;
		double resi = 1.0;
		double initres = 1.0;
		Matrix<double>* err;
		err = new Matrix<double>[nvars];
		for(int i = 0; i<nvars; i++)
			err[i].setup(m->gnelem(),1);
		Matrix<double> res(nvars,1);
		res.ones();
		Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);
		Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
		Matrix<double> uoldest(u.rows(), u.cols(), ROWMAJOR);
		Matrix<double> R = r_domain + r_boundary;

		calculate_leastsquaresLHS();

		while(resi/initres > tol)// || step <= 10)
		{
			uoldest = u;

			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			integ.zeros();		// reset CFL data

			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();		// this invokes Flux calculating function

			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) += dtm(iel)*m_inverse.get(iel)*R.get(iel,i);
				}
			}

			//----------- stage 2 ------------
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			integ.zeros();		// reset CFL data

			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();		// this invokes Flux calculating function

			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) = 0.75*uoldest(iel,i) + 0.25*(uold(iel,i) + dtm(iel)*m_inverse.get(iel)*R.get(iel,i));
				}
			}

			// ------------- stage 3 ----------------
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			integ.zeros();		// reset CFL data

			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();		// this invokes Flux calculating function

			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) = 1.0/3*uoldest(iel,i) + 2.0/3*(uold(iel,i) + dtm(iel)*m_inverse.get(iel)*R.get(iel,i));
				}
			}

			for(int i = 0; i < nvars; i++)
			{
				err[i] = (u-uold).col(i);
				res(i) = l2norm(&err[i]);
			}
			resi = res.max();

			if(step == 0)
				initres = resi;

			if(step % 10 == 0)
				cout << "EulerFV: solve_rk1_steady(): Step " << step << ", rel residual " << resi/initres << endl;

			step++;
			/*double totalenergy = 0;
			for(int i = 0; i < m->gnelem(); i++)
				totalenergy += u(i,3)*m->jacobians(i);
			cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
			//if(step == 10000) break;
		}

		//calculate gradients
		flux.compute_ls_reconstruction();
		delete [] err;
	}

	//Post-processing function - produces scalar and vector arrays for output to VTU
	void postprocess()
	{
		cout << "EulerFV: postprocess(): Preparing output matrices...\n";
		scalars.setup(m->gnpoin(), 3, COLMAJOR);
		velocities.setup(m->gnpoin(), 2, ROWMAJOR);
		Matrix<double> c(m->gnpoin(), 1, ROWMAJOR);

		Matrix<double> xi = flux.centroid_x_coords();
		Matrix<double> yi = flux.centroid_y_coords();

		//Create nodal variables array
		Matrix<double> un(m->gnpoin(),nvars,ROWMAJOR);
		//un.zeros();

		cout << "EulerFV: postprocess(): Calculating nodal values...\n";
		//For each point, take area-weighted average of values at that point from all elements surrounding it
		/*for(int ip = 0; ip < m->gnpoin(); ip++)
		{
			double sigw = 0;		// sum of area weights
			for(int i = m->gesup_p(ip); i <= m->gesup_p(ip+1)-1; i++)
			{
				int elem = m->gesup(i);
				Matrix<double> val(nvars,1,ROWMAJOR);
				for(int j = 0; j < nvars; j++)
				{
					val(j) = u(elem,j) + dudx(elem,j)*(m->gcoords(ip,0)-xi(elem)) + dudy(elem,j)*(m->gcoords(ip,1)-yi(elem));
					un(ip,j) += val(j)*m->jacobians(elem);
				}
				sigw += m->jacobians(elem);
			}
			for(int j = 0; j < nvars; j++) {
				un(ip,j) /= sigw;
			}
		}*/

		// For each element take simple average of values around it
		for(int ip = 0; ip < m->gnpoin(); ip++)
		{
			int elem_surr = 0;		// keep track of number of elements surrounding this node
			for(int i = m->gesup_p(ip); i <= m->gesup_p(ip+1)-1; i++)
			{
				int elem = m->gesup(i);
				Matrix<double> val(nvars,1,ROWMAJOR);
				for(int j = 0; j < nvars; j++)
				{
					val(j) = u(elem,j) + dudx(elem,j)*(m->gcoords(ip,0)-xi(elem)) + dudy(elem,j)*(m->gcoords(ip,1)-yi(elem));
					un(ip,j) += val(j);
				}
				elem_surr++;
			}
			for(int j = 0; j < nvars; j++) {
				un(ip,j) /= elem_surr;
			}
		}

		cout << "EulerFV: postprocess(): Preparing output matrices\n";
		Matrix<double> d = un.col(0);
		scalars.replacecol(0, d);		// populate density data
		//cout << "EulerFV: postprocess(): Written density\n";

		for(int i = 0; i < m->gnpoin(); i++)
		{
			velocities(i,0) = un(i,1)/un(i,0);
			velocities(i,1) = un(i,2)/un(i,0);
			double vmag2 = pow(velocities(i,0), 2) + pow(velocities(i,1), 2);
			scalars(i,2) = d(i)*(g-1) * (un(i,3)/d(i) - 0.5*vmag2);		// pressure
			c(i) = sqrt(g*scalars(i,2)/d(i));
			scalars(i,1) = sqrt(vmag2)/c(i);
		}
		cout << "EulerFV: postprocess(): Done.\n";
	}

	void postprocess_cell()
	{
		cout << "EulerFV: postprocess_cell(): Creating output matrices...\n";
		scalars.setup(m->gnelem(), 3, COLMAJOR);
		velocities.setup(m->gnelem(), 2, ROWMAJOR);
		Matrix<double> c(m->gnelem(), 1, ROWMAJOR);

		cout << "EulerFV: postprocess(): Populating output matrices\n";

		Matrix<double> d = u.col(0);
		scalars.replacecol(0, d);		// populate density data
		//cout << "EulerFV: postprocess(): Written density\n";

		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			velocities(iel,0) = u(iel,1)/u(iel,0);
			velocities(iel,1) = u(iel,2)/u(iel,0);
			//velocities(iel,0) = dudx(iel,1);
			//velocities(iel,1) = dudy(iel,1);
			double vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
			scalars(iel,2) = d(iel)*(g-1) * (u(iel,3)/d(iel) - 0.5*vmag2);		// pressure
			c(iel) = sqrt(g*scalars(iel,2)/d(iel));
			scalars(iel,1) = sqrt(vmag2)/c(iel);
		}
		cout << "EulerFV: postprocess_cell(): Done.\n";
	}

	double compute_error_cell()		// call after postprocess_cell
	{
		double vmaginf2 = uinf(0,1)/uinf(0,0)*uinf(0,1)/uinf(0,0) + uinf(0,2)/uinf(0,0)*uinf(0,2)/uinf(0,0);
		double sinf = ( uinf(0,0)*(g-1) * (uinf(0,3)/uinf(0,0) - 0.5*vmaginf2) ) / pow(uinf(0,0),g);

		Matrix<double> s_err(m->gnelem(),1,ROWMAJOR);
		double error = 0;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			s_err(iel) = (scalars(iel,2)/pow(scalars(iel,0),g) - sinf)/sinf;
			error += s_err(iel)*s_err(iel)*m->jacobians(iel)/2.0;
		}
		error = sqrt(error);

		//double h = (m->jacobians()).min();
		double h = 1/sqrt(m->gnelem());

		cout << "EulerFV:   " << log(h) << "  " << log(error) << endl;

		return error;
	}
};

} // end namespace acfd

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
#include "atimeint.hpp"

using namespace std;
using namespace amat;
using namespace acfd;

namespace acfd {

// set the flux calculation method here
typedef VanLeerFlux1 Flux;
//set time explicit stepping scheme here
typedef TimeStepRK1 TimeStep;

class EulerFV
{
	UTriMesh* m;
	Matrix<double> m_inverse;
	Matrix<double> r_domain;
	Matrix<double> r_boundary;
	int nvars;
	Matrix<double> uinf;
	Matrix<double> integ;		// stores int_{\partial \Omega_I} ( |v_n| + c) d \Gamma, where v_n and c are average values for each face

	Flux flux;

public:
	Matrix<double> u;			// vector of unknowns
	// The following 2 are for postprocessing
	Matrix<double> scalars;		// col 1 contains density, col 2 contains mach number, col 3 contains pressure
	Matrix<double> velocities;

	EulerFV(UTriMesh* mesh)		// Make sure jacobians and face data are computed!
	{
		m = mesh;
		nvars = 4;

		m_inverse.setup(m->gnelem(),1,ROWMAJOR);		// just a vector for FVM. For DG, this will be an array of Matrices
		r_domain.setup(m->gnelem(),nvars,ROWMAJOR);
		r_boundary.setup(m->gnelem(),nvars,ROWMAJOR);
		u.setup(m->gnelem(), nvars, ROWMAJOR);
		uinf.setup(1, nvars, ROWMAJOR);
		integ.setup(m->gnelem(), 1,ROWMAJOR);

		for(int i = 0; i < m->gnelem(); i++)
			m_inverse(i) = 2.0/mesh->jacobians(i);

		flux.setup(m, &u, &uinf, &r_boundary, &integ);
	}

	void loaddata(double Minf, double vinf, double a, double rhoinf)
	{
		// load initial data. Note that a is in radians
		// Note that reference density and reference velocity are the values at infinity
		cout << "EulerFV: loaddata(): Calculating initial data...\n";
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
		cout << "EulerFV: loaddata(): Initial data calculated.\n";
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

	void solve(double time, double cfl)
	{
		double telapsed = 0.0;
		int step = 0;

		while(telapsed <= time)
		{
			cout << "EulerFV: solve(): Step " << step << ", time elapsed " << telapsed << endl;

			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();
			//cout << "Integ check: " << integ(6,0) << endl;

			//calculate dt based on CFL
			// Vector to hold time steps
			Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);

			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			// Finally get global time step
			double dt = dtm.min();
			cout << "EulerFV: solve(): Current time step = " << dt << endl;

			TimeStep ts(m, &m_inverse, &r_domain, &r_boundary, &u, dt);
			// advance solution 1 step in time
			ts.advance();

			integ.zeros();		// reset CFL data
			telapsed += dt;
			step++;
			/*double totalenergy = 0;
			for(int i = 0; i < m->gnelem(); i++)
				totalenergy += u(i,3)*m->jacobians(i);
			cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
		}
	}

	void solve_steady(double tol, double cfl)
	{
		double telapsed = 0.0;
		int step = 0;
		double resi = 1.0;
		double initres = 1.0;
		Matrix<double> res(nvars,1);
		res.ones();
		Matrix<double>* err;
		err = new Matrix<double>[nvars];
		for(int i = 0; i<nvars; i++)
			err[i].setup(m->gnelem(),1);

		while(resi/initres > tol)
		{

			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();
			//cout << "Integ check: " << integ(6,0) << endl;

			//calculate dt based on CFL
			// Vector to hold time steps
			Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);

			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			//cout << "EulerFV: solve(): Current time step = " << dt << endl;

			Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
			Matrix<double> R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) += dtm(iel)*m_inverse.get(iel)*R.get(iel,i);
				}
			}

			integ.zeros();		// reset CFL data


			for(int i = 0; i < nvars; i++)
			{
				err[i] = (u-uold).col(i);
				res(i) = l2norm(&err[i]);
			}
			resi = res.max();
			if (step == 0) initres = resi;

			if(step % 10 == 0)
				cout << "EulerFV: solve(): Step " << step << ", rel residual " << resi/initres << endl;

			step++;
			//if(step == 10000) break;
			/*double totalenergy = 0;
			for(int i = 0; i < m->gnelem(); i++)
				totalenergy += u(i,3)*m->jacobians(i);
			cout << "EulerFV: solve(): Total energy = " << totalenergy << endl;*/
		}
		delete [] err;
	}

	void solve_steady_rk3(double tol, double cfl)
	{
		double telapsed = 0.0;
		int step = 0;
		double resi = 1.0;
		double initres = 1.0;
		Matrix<double> res(nvars,1);
		res.ones();
		Matrix<double>* err;
		err = new Matrix<double>[nvars];
		for(int i = 0; i<nvars; i++)
			err[i].setup(m->gnelem(),1);

		Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
		Matrix<double> uoldest(u.rows(), u.cols(), ROWMAJOR);
		Matrix<double> dtm(m->gnelem(), 1, ROWMAJOR);
		Matrix<double> R = r_domain + r_boundary;

		while(resi/initres > tol)
		{
			uoldest = u;
			
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();
			//cout << "Integ check: " << integ(6,0) << endl;

			//calculate dt based on CFL
			// Vector to hold time steps

			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm(iel) = cfl*(0.5*m->gjacobians(iel)/integ(iel));
			}

			//cout << "EulerFV: solve(): Current time step = " << dt << endl;
			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) += dtm(iel)*m_inverse.get(iel)*R.get(iel,i);
				}
			}

			integ.zeros();		// reset CFL data

			// -------------- second stage -------------------
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();
			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) = 0.75*uoldest(iel,i) + 0.25*(uold(iel,i) + dtm(iel)*m_inverse.get(iel)*R.get(iel,i));
				}
			}

			integ.zeros();		// reset CFL data

			// --------------------- third stage ---------------------
			// reset fluxes
			r_domain.zeros();
			r_boundary.zeros();
			//calculate fluxes
			calculate_r_domain();
			calculate_r_boundary();

			Matrix<double> uold(u.rows(), u.cols(), ROWMAJOR);
			R = r_domain + r_boundary;
			uold = u;
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				for(int i = 0; i < nvars; i++)
				{
					u(iel,i) = 1.0/3*uoldest(iel,i) + 2.0/3*(uold(iel,i) + dtm(iel)*m_inverse.get(iel)*R.get(iel,i));
				}
			}

			integ.zeros();		// reset CFL data


			for(int i = 0; i < nvars; i++)
			{
				err[i] = (u-uold).col(i);
				res(i) = l2norm(&err[i]);
			}
			resi = res.max();
			if (step == 0) initres = resi;

			if(step % 10 == 0)
				cout << "EulerFV: solve(): Step " << step << ", rel residual " << resi/initres << endl;

			step++;
		}
		delete [] err;
	}

	//Post-processing function - produces scalar and vector arrays for output to VTU
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
			double vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
			scalars(iel,2) = d(iel)*(g-1) * (u(iel,3)/d(iel) - 0.5*vmag2);		// pressure
			c(iel) = sqrt(g*scalars(iel,2)/d(iel));
			scalars(iel,1) = sqrt(vmag2)/c(iel);
		}
		cout << "EulerFV: postprocess_cell(): Done.\n";
	}

	void postprocess()
	{
		cout << "EulerFV: postprocess(): Creating output matrices...\n";
		scalars.setup(m->gnpoin(), 3, COLMAJOR);
		velocities.setup(m->gnpoin(), 2, ROWMAJOR);
		Matrix<double> c(m->gnpoin(), 1, ROWMAJOR);

		//Create nodal variables array
		Matrix<double> un(m->gnpoin(),nvars,ROWMAJOR);

		cout << "EulerFV: postprocess(): Populating output matrices\n";


		//cout << "EulerFV: postprocess(): Written density\n";

		//For each point, take area-weighted average of values at that point from all elements surrounding it
		for(int ip = 0; ip < m->gnpoin(); ip++)
		{
			double sigw = 0;		// sum of area weights
			for(int i = m->gesup_p(ip); i <= m->gesup_p(ip+1)-1; i++)
			{
				int elem = m->gesup(i);
				for(int j = 0; j < nvars; j++)
				{
					un(ip,j) += u(elem,j)*m->jacobians(elem);
				}
				sigw += m->jacobians(elem);
			}
			for(int j = 0; j < nvars; j++) {
				un(ip,j) /= sigw;
			}
		}
		Matrix<double> d = un.col(0);
		scalars.replacecol(0, d);		// populate density data
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

	double compute_error_cell()		// call after postprocess_cell
	{
		double vmaginf2 = uinf(0,1)/uinf(0,0)*uinf(0,1)/uinf(0,0) + uinf(0,2)/uinf(0,0)*uinf(0,2)/uinf(0,0);
		double sinf = ( uinf(0,0)*(g-1) * (uinf(0,3)/uinf(0,0) - 0.5*vmaginf2) ) / pow(uinf(0,0),g);

		Matrix<double> s_err(m->gnelem(),1,ROWMAJOR);
		double error = 0; double area = 0;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			s_err(iel) = (scalars(iel,2)/pow(scalars(iel,0),g) - sinf)/sinf;
			error += s_err(iel)*s_err(iel)*m->jacobians(iel)/2.0;
			//error += s_err(iel)*m->jacobians(iel);
			//area += m->jacobians(iel);
		}
		error = sqrt(error);
		//error = error/area;
		//double h = sqrt((m->jacobians).min());
		double h = 1/sqrt(m->gnelem());

		cout << "EulerFV:   " << log(h) << "  " << log(error) << endl;

		return error;
	}
};

} // end namespace acfd

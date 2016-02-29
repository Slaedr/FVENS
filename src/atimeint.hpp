/** Performs one time step of Matrix ODE of the form M dU/dx = R (where M is a NxN matrix while U and R are 
N-vectors) in an EXPLICIT scheme. */

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __AMESH2D_H
#include <amesh2.hpp>
#endif

#define __ATIMEINT_H

using namespace amat;
using namespace acfd;

namespace acfd {

// Performs 1 step of time integration for all cells
class TimeStepRK1
{
	UTriMesh* m;
	Matrix<double>* M;
	Matrix<double>* R;
	Matrix<double>* u;
	int nvars;
	double dt;
public:
	TimeStepRK1(UTriMesh* mesh, Matrix<double>* M_, Matrix<double>* r_dom, Matrix<double>* r_boun, Matrix<double>* unknowns, double deltat)
	{
		m = mesh;
		M = M_;			// inverse of mass "matrix"
		
		R = new Matrix<double>;
		R->setup(r_boun->rows(), r_boun->cols(), ROWMAJOR);
		*R = (*r_dom) + (*r_boun);		// Matrix *R now holds the total RHS
		//R->mprint();
		
		u = unknowns;
		nvars = u->cols();
		dt = deltat;
	}
	
	~TimeStepRK1()
	{
		delete R;
	}
	
	void advance()
	{
		//Matrix<double> uold(u->rows(), u->cols(), ROWMAJOR);
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
			{
				u->operator()(iel,i) += dt*M->get(iel)*R->get(iel,i);
			}
		}
	}
};

class TimeStepRK2
{
	UTriMesh* m;
	Matrix<double>* M;
	Matrix<double>* R;
	Matrix<double>* u;
	int nvars;
	double dt;
public:
	TimeStepRK2(UTriMesh* mesh, Matrix<double>* M_, Matrix<double>* r_dom, Matrix<double>* r_boun, Matrix<double>* unknowns, double deltat)
	{
		m = mesh;
		M = M_;			// inverse of mass "matrix"
		
		R = new Matrix<double>;
		R->setup(r_boun->rows(), r_boun->cols(), ROWMAJOR);
		*R = (*r_dom) + (*r_boun);		// Matrix *R now holds the total RHS
		//R->mprint();
		
		u = unknowns;
		nvars = u->cols();
		dt = deltat;
	}
	
	~TimeStepRK2()
	{
		delete R;
	}
	
	void advance()
	{
		//Matrix<double> uold(u->rows(), u->cols(), ROWMAJOR);
		//uold = *u;
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
			{
				u->operator()(iel,i) += dt*M->get(iel)*R->get(iel,i);
			}
		}
	}
};

} // end namespace acfd

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#define __AQUADRATURE2D_H

using namespace amat;

namespace acfd {

const int ndimn = 2;

//The Quadrature2D classes calculate flux integrals over ONE face

class Quadrature2D		// generic quadrature class - not ready yet
{
	double xi;
	double yi;
	double xe;
	double ye;
	double order;
	int nvars;
	Matrix<double> gpoints;		//array of Gauss points
	Matrix<double> fpoints;		//values of flux at the Guass points
	Matrix<double>* integral;	// This points to storage of final integrated values
	
public:
	Quadrature2D(double xii, double xee, double yii, double yee, double orderr, double number_variables, Matrix<double>* integral_in)
	{
		xi = xii; xe = xee; yi = yii; ye = yee;
		order = orderr;
		nvars = number_variables;
		gpoints.setup(order, ndimn, COLMAJOR);
		fpoints.setup(order, nvars, COLMAJOR);
		integral = integral_in;
	}
	
	Matrix<double> gintegral()
	{
	}
};

class LinearQuadrature2D
{
	int nvars;
	Matrix<double>* flux;
	Matrix<double>* integral;
public:
	LinearQuadrature2D(Matrix<double>* fluxx, Matrix<double>* integral_in)
	{
		flux = fluxx;
		integral = integral_in;
		nvars = flux->cols();
	}
	
	void gintegral()
	{
		// Simply copy data in flux to integral
	}	
};

} // end namespace acfd
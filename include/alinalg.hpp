/* A library to solve linear systems.
   Aditya Kashi
   Feb 2015
*/

#ifndef __AMATRIX2_H
#include <amatrix2.hpp>
#endif

#ifndef __ASPARSEMATRIX_H
#include <asparsematrix.hpp>
#endif

#ifndef _GLIBCXX_CMATH
#include <cmath>
#endif

#ifdef _OPENMP
#ifndef OMP_H
#include <omp.h>
#define nthreads_linalg 8
#endif
#endif

#define __ALINALG_H 1

using namespace std;
using namespace amat;

namespace amat
{

typedef MatrixCOO SpMatrix;

/* Note: Cholesky algorithm only implemented for a row-major matrix */
Matrix<double> cholesky(Matrix<double> A, Matrix<double> b)
{
	Matrix<double> B;

	cout << "\ncholesky: Input LHS matrix is " << A.rows() << " x " << A.cols() << endl;
	if(A.rows() != b.rows()) { cout << "\nInvalid dimensions of A and b!"; return B; }
	int N = A.rows();

	//Part 1: Cholesky decomposition
	B.setup(N,N,ROWMAJOR); B.zeros();

	B(0,0) = sqrt(A(0,0));
	for(int i = 1; i < N; i++)
		B(i,0) = A(i,0)/B(0,0);

	for(int j = 1; j < N; j++)
	{
		double bjk_sum = 0;
		int k = 0;
		do
		{
			bjk_sum += B(j,k)*B(j,k);
			k++;
		}
		while(k <= j-1);

		if(bjk_sum >= A(j,j)) cout << "\n! cholesky: Negative argument to sqrt at ("<<j<<","<<j<<")\n";
		B(j,j) = sqrt(A(j,j) - bjk_sum);

		for(int i = j+1; i < N; i++)
		{
			double bsum = 0;
			k=0;
			do
			{	bsum += B(i,k)*B(j,k);
				k++;
			}
			while(k <= j-1);
			B(i,j) = (A(i,j) - bsum)/B(j,j);
		}
	}
	// We now have B, the lower triangular matrix

	// Check if any of the diagonal elements of B are zero
	for(int i = 0; i < N; i++)
		if(abs(B(i,i)) < 1e-10)
		{
			cout << "\ncholesky: Element (" << i <<"," << i << ") of lower triangular matrix is near zero!";
			return B;
		}

	// Part 2: forward substitution to obtain intermediate vector y
	Matrix<double> y(N,1,ROWMAJOR);

	y(0,0) = b(0,0)/B(0,0);

	for(int i = 1; i < N; i++)
	{
		double sum = 0;
		int k = 0;
		do
		{	sum += B(i,k)*y(k,0);
			k++;
		} while(k <= i-1);
		y(i,0) = (b(i,0) - sum)/B(i,i);
	}

	//Part 3: back substitution to obtain final solution
	// Note: the final solution is stored in b
	b.zeros();
	//Matrix<double> f(N,1,ROWMAJOR);
	b(N-1,0) = y(N-1,0)/B(N-1,N-1);

	for(int i = N-2; i >= 0; i--)
	{
		double sum = 0;
		int k = i+1;
		do
		{	sum += B(k,i)*b(k,0);
			k++;
		} while(k <= N-1);
		b(i,0) = (y(i,0) - sum)/B(i,i);
	}

	return b;
}

Matrix<double> gausselim(Matrix<double> A, Matrix<double> b, double tol=1e-12)
{
	//cout << "gausselim: Input LHS matrix is " << A.rows() << " x " << A.cols() << endl;
	if(A.rows() != b.rows()) { cout << "gausselim: Invalid dimensions of A and b!\n"; return A; }
	int N = A.rows();

	Matrix<double> x(N,1);
	x.zeros();

	for(int i = 0; i < N-1; i++)
	{
		double max = dabs(A(i,i));
		int maxr = i;
		for(int j = i+1; j < N; j++)
		{
			if(dabs(A(j,i)) > max)
			{
				max = dabs(A(j,i));
				maxr = j;
			}
		}
		if(max > tol)
		{
			//interchange rows i and maxr 
			for(int k = i; k < N; k++)
			{
				double temp = A(i,k);
				A(i,k) = A(maxr,k);
				A(maxr,k) = temp;
			}
			// do the interchange for b as well
			double temp = b(i,0);
			b(i,0) = b(maxr,0);
			b(maxr,0) = temp;
		}
		else { cout << "! gausselim: Pivot not found!!\n"; return x; }

		for(int j = i+1; j < N; j++)
		{
			double ff = A(j,i);
			for(int l = i; l < N; l++)
				A(j,l) = A(j,l) - ff/A(i,i)*A(i,l);
			b(j,0) = b(j,0) - ff/A(i,i)*b(i,0);
		}
	}
	//Thus, A has been transformed to an upper triangular matrix, b has been transformed accordingly.

	//Part 2: back substitution to obtain final solution
	// Note: the solution is stored in x
	//Matrix<double> f(N,1,ROWMAJOR);
	x(N-1,0) = b(N-1,0)/A(N-1,N-1);

	for(int i = N-2; i >= 0; i--)
	{
		double sum = 0;
		int k = i+1;
		do
		{	sum += A(i,k)*x(k,0);		// or A(k,i) ??
			k++;
		} while(k <= N-1);
		x(i,0) = (b(i,0) - sum)/A(i,i);
	}
	return x;
}


//-------------------- Iterative Methods ----------------------------------------//

Matrix<double> pointjacobi(Matrix<double> A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter, char check)
{
	cout << "\npointjacobi: Input LHS matrix is " << A.rows() << " x " << A.cols() << endl;
	if(A.rows() != b.rows()) { cout << "! pointjacobi: Invalid dimensions of A and b!\n"; return b; }
	if(xold.rows() != b.rows()) { cout << "! pointjacobi: Invalid dimensions on xold !\n"; return b; }
	int N = A.rows();

	if(check == 'y')
	{
		for(int i = 0; i < A.rows(); i++)
		{
			double sum = 0;
			for(int j = 0; j < A.cols(); j++)
			{
				if(i != j) sum += dabs(A(i,j));
			}
			if(dabs(A(i,i)) <= sum) cout << "* pointjacobi(): LHS Matrix is NOT strictly diagonally dominant in row " << i << "!!\n";
		}
	}

	Matrix<double> x(b.rows(),1);

	Matrix<double> M(N,N,ROWMAJOR);		// diagonal matrix
	M.zeros();

	//populate diagonal matrix M
	for(int i = 0; i < N; i++)
		M(i,i) = A(i,i);

	A = A - M;

	//invert M
	for(int i = 0; i < N; i++)
		M(i,i) = 1.0/M(i,i);

	//x.zeros();
	//cout << "pointjacobi: Initial error = " << (xold-x).dabsmax() << endl;

	x = xold;
	int c = 0;

	do
	{
		xold = x;
		x = M * (b - (A*xold));
		c++;
		if(c > maxiter) { cout << "pointjacobi: Max iterations exceeded!\n"; break; }
	} while((x-xold).dabsmax() >= tol);

	/* double error = 1.0;
	do
	{
		xold = x;

		Matrix<double> Axold(N,1);
		Axold.zeros();
		for(int i = 0; i < N; i++)
		{
			for(int k = 0; k < N; k++)
				Axold(i,0) += A(i,k)*xold(k,0);
		}
		Matrix<double> inter(N,1);
		for(int i = 0; i < N; i++)
			inter(i,0) = b(i,0) - Axold(i,0);
		for(int i = 0; i < N; i++)
			x(i,0) = M(i,i) * inter(i,0);

		c++;
		if(c > maxiter) { cout << "pointjacobi: Max iterations exceeded!\n"; break; }
		Matrix<double> diff(N,1);	// diff = x - xold
		for(int i = 0; i < N; i++)
			diff(i,0) = x(i,0) - xold(i,0);

		error = dabs(diff(0,0));
		for(int i = 1; i < N; i++)
			if(dabs(diff(i,0)) > error) error = dabs(diff(i,0));
		//cout << "pointjacobi: error = " << error << endl;

	} while(error >= tol); */
	cout << "pointjacobi: No. of iterations = " << c << endl;

	return x;
}

Matrix<double> gaussseidel(Matrix<double> A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter, char check='n')
{
	//cout << "\ngaussseidel: Input LHS matrix is " << A.rows() << " x " << A.cols() << endl;
	if(A.rows() != b.rows()) { cout << "! gaussseidel: Invalid dimensions of A and b!\n"; return b; }
	if(xold.rows() != b.rows()) { cout << "! gaussseidel: Invalid dimensions on xold !\n"; return b; }
	int N = A.rows();

	if(check == 'y')
	{
		for(int i = 0; i < A.rows(); i++)
		{
			double sum = 0;
			for(int j = 0; j < A.cols(); j++)
			{
				if(i != j) sum += dabs(A(i,j));
			}
			if(dabs(A(i,i)) <= sum) cout << "* gaussseidel: LHS Matrix is NOT strictly diagonally dominant in row " << i << "!!\n";
		}
	}

	Matrix<double> x(b.rows(),1);

	Matrix<double> M(N,N,ROWMAJOR);		// diagonal matrix
	M.zeros();

	//populate diagonal matrix M
	for(int i = 0; i < N; i++)
		M(i,i) = A(i,i);

	A = A - M;						// diagonal entries of A are now zero

	//invert M
	for(int i = 0; i < N; i++)
		M(i,i) = 1.0/M(i,i);

	//x.zeros();
	//cout << "gaussseidel: Initial error = " << (xold-x).dabsmax() << endl;

	x = xold;
	int c = 0;
	double initres;
	bool first = true;

	Matrix<double> Axold(N,1);
	Matrix<double> inter(N,1);
	Matrix<double> diff(N,1);	// diff = x - xold
	double error = 1.0;
	do
	{
		xold = x;
		Axold.zeros();
		for(int i = 0; i < N; i++)
		{
			for(int k = 0; k < i; k++)
				Axold(i,0) += A(i,k)*x(k,0);
			for(int k = i; k < N; k++)
				Axold(i,0) += A(i,k)*xold(k,0);
			inter(i,0) = b(i,0) - Axold(i,0);
			x(i,0) = M(i,i) * inter(i,0);
		}
		// NOTE: The above results in FORWARD Gauss-Seidel

		c++;
		if(c > maxiter) { cout << "gaussseidel: Max iterations exceeded!\n"; break; }
		for(int i = 0; i < N; i++)
			diff(i,0) = x(i,0) - xold(i,0);

		error = dabs(diff(0,0));
		for(int i = 1; i < N; i++)
			if(dabs(diff(i,0)) > error) error = dabs(diff(i,0));
		//cout << "gaussseidel: error = " << error << endl;
		if(first == true)
		{	initres = error;
			//cout << "gaussseidel: Initial residue = " << initres << endl;
			first = false;
		}

		//if(c%20 == 0) cout << "gaussseidel(): Step " << c << ", Relative error = " << error/initres << endl;

	} while(error/initres >= tol);
	//cout << "gaussseidel: No. of iterations = " << c << endl;

	return x;
}

Matrix<double> sparsegaussseidel(SpMatrix* A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter, char check='n')
{
	cout << "sparsegaussseidel(): Input LHS matrix is " << A->rows() << " x " << A->cols() << endl;
	cout << "sparsegaussseidel(): b is " << b.rows() << ", and xold is " << xold.rows() << endl;
	if(A->rows() != b.rows()) { cout << "! gaussseidel: Invalid dimensions of A and b!\n"; return b; }
	if(xold.rows() != b.rows()) { cout << "! gaussseidel: Invalid dimensions on xold !\n"; return b; }
	int N = A->rows();

	if(check == 'y')
	{
		for(int i = 0; i < A->rows(); i++)
		{
			double sum = 0;
			for(int j = 0; j < A->cols(); j++)
			{
				if(i != j) sum += dabs(A->get(i,j));
			}
			if(dabs(A->get(i,i)) <= sum) cout << "* gaussseidel: LHS Matrix is NOT strictly diagonally dominant in row " << i << "!!\n";
		}
	}

	Matrix<double> x(b.rows(),1);

	Matrix<double> M(N,1);		// vector of diagonal elements of A
	M.zeros();

	// diagonal matrix M
	//cout << "sparsegaussseidel(): Getting diagonal of sparse matrix\n";
	A->get_diagonal(&M);

	//invert M
	for(int i = 0; i < N; i++)
		M(i) = 1.0/M(i);

	//x.zeros();
	//cout << "gaussseidel: Initial error = " << (xold-x).dabsmax() << endl;

	for(int i = 0; i < x.rows(); i++)
		x(i) = xold(i);
	int c = 0;
	double initres;
	bool first = true;

	Matrix<double> Axold(N,1);
	Matrix<double> inter(N,1);
	Matrix<double> diff(N,1);	// diff = x - xold
	double error = 1.0;
	//cout << "sparsegaussseidel(): Starting iterations\n";
	do
	{
		xold = x;
		Axold.zeros();
		int i;

		//#pragma omp parallel for default(none) private(i) shared(A,b,Axold,x,xold,inter,M,N) num_threads(nthreads_linalg)
		for(i = 0; i < N; i++)
		{
			/*for(int k = 0; k < i; k++)
				Axold(i,0) += A(i,k)*x(k,0);
			for(int k = i; k < N; k++)
				Axold(i,0) += A(i,k)*xold(k,0);*/

			//cout << "  Calling sparse getelem_multiply_parts\n";
			Axold(i) = A->getelem_multiply_parts(i, &x, &xold, i, xold.get(i));
			//cout << "  Setting inter\n";
			inter(i,0) = b(i,0) - Axold(i,0);
			x(i,0) = M(i) * inter(i,0);
		}
		// NOTE: The above results in FORWARD Gauss-Seidel

		//if(c > maxiter) { cout << "gaussseidel: Max iterations exceeded!\n"; break; }
		for(int i = 0; i < N; i++)
			diff(i,0) = x(i,0) - xold(i,0);

		error = dabs(diff(0,0));
		for(int i = 1; i < N; i++)
			if(dabs(diff(i,0)) > error) error = dabs(diff(i,0));
		//cout << "gaussseidel: error = " << error << endl;
		if(first == true)
		{	initres = error;
			if(dabs(initres) < tol*tol)
			{
				cout << "sparsegaussseidel(): Initial residue = " << initres << endl;
				break;
			}
			first = false;
		}

		if(c%10 == 0 || c == 1) cout << "gaussseidel(): Step " << c << ", Relative error = " << error/initres << endl;
		c++;

	} while(error/initres >= tol && c <= maxiter);
	cout << "gaussseidel: No. of iterations = " << c << ", final error " << error/initres << endl;

	return x;
}

Matrix<double> sparseSOR(SpMatrix* A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter, double w=1.25, char check='n')
// CAUTION: does not work in parallel due to some reason
{
	cout << "sparseSOR(): Input LHS matrix is " << A->rows() << " x " << A->cols() << endl;
	if(A->rows() != b.rows()) { cout << "! sparseSOR(): Invalid dimensions of A and b!\n"; return b; }
	if(xold.rows() != b.rows()) { cout << "! sparseSOR(): Invalid dimensions on xold !\n"; return b; }
	int N = A->rows();

	if(check == 'y')
	{
		for(int i = 0; i < A->rows(); i++)
		{
			double sum = 0;
			for(int j = 0; j < A->cols(); j++)
			{
				if(i != j) sum += dabs(A->get(i,j));
			}
			if(dabs(A->get(i,i)) <= sum) cout << "* sparseSOR(): LHS Matrix is NOT strictly diagonally dominant in row " << i << "!!\n";
		}
	}

	Matrix<double> x(b.rows(),1);

	Matrix<double> M(N,1);		// vector of diagonal elements of A
	M.zeros();

	// diagonal matrix M
	//cout << "sparsegaussseidel(): Getting diagonal of sparse matrix\n";
	A->get_diagonal(&M);

	//invert M
	for(int i = 0; i < N; i++)
		M(i) = 1.0/M(i);

	//x.zeros();
	//cout << "gaussseidel: Initial error = " << (xold-x).dabsmax() << endl;

	x = xold;
	int c = 0;
	double initres;
	bool first = true;

	Matrix<double> Axold(N,1);
	Matrix<double> inter(N,1);
	Matrix<double> diff(N,1);	// diff = x - xold
	double error = 1.0;
	//cout << "sparsegaussseidel(): Starting iterations\n";
	do
	{
		xold = x;
		Axold.zeros();
		int i;

		//#pragma omp parallel for default(none) private(i) shared(A,b,Axold,x,xold,inter,M,N,w) num_threads(nthreads_linalg)
		for(i = 0; i < N; i++)
		{
			//cout << "  Calling sparse getelem_multiply_parts\n";
			Axold(i) = A->getelem_multiply_parts(i, &x, &xold, i, xold.get(i));
			//cout << "  Setting inter\n";
			inter(i,0) = w*(b(i,0) - Axold(i,0));
			x(i,0) = (1-w)*xold(i,0) + M(i) * inter(i,0);
		}
		// NOTE: The above results in FORWARD Gauss-Seidel

		//if(c > maxiter) { cout << "gaussseidel: Max iterations exceeded!\n"; break; }
		for(int i = 0; i < N; i++)
			diff(i,0) = x(i,0) - xold(i,0);

		error = dabs(diff(0,0));
		for(int i = 1; i < N; i++)
			if(dabs(diff(i,0)) > error) error = dabs(diff(i,0));
		//cout << "gaussseidel: error = " << error << endl;
		if(first == true)
		{	initres = error;
			if(dabs(initres) < tol*tol)
			{
				cout << "sparseSOR(): Initial residue = " << initres << endl;
				break;
			}
			first = false;
		}

		if(c%10 == 0 || c == 1) cout << "sparseSOR(): Step " << c << ", Relative error = " << error/initres << endl;
		c++;

	} while(error/initres >= tol && c <= maxiter);
	cout << "sparseSOR(): No. of iterations = " << c << ", final error " << error/initres << endl;

	return x;
}

Matrix<double> sparseCG_d(SpMatrix* A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter)
/* Calculates solution of Ax=b where A is a SPD matrix in sparse format. The preconditioner is a diagonal matrix.*/
{
	cout << "sparseCG_d(): Solving " << A->rows() << "x" << A->cols() << " system by conjugate gradient method with diagonal preconditioner\n";

	// check
	//if(A->rows() != b.rows() || A->rows() != xold.rows()) cout << "sparseCG_d(): ! Mismatch in number of rows!!" << endl;

	Matrix<double> x(A->rows(),1);		// solution vector
	Matrix<double> M(A->rows(), 1);		// diagonal preconditioner, or soon, inverse of preconditioner
	Matrix<double> rold(A->rows(),1);		// initial residual = b - A*xold
	Matrix<double> r(A->rows(),1);			// residual = b - A*x
	Matrix<double> z(A->rows(),1);
	Matrix<double> zold(A->rows(),1);
	Matrix<double> p(A->rows(),1);
	Matrix<double> pold(A->rows(),1);
	Matrix<double> temp(A->rows(),1);
	Matrix<double> diff(A->rows(),1);
	double temp1, temp2;
	double theta;
	double beta;
	double error = 1.0;
	double initres;
	double normalizer = b.l2norm();

	//cout << "sparseCG_d(): Declared everything" << endl;

	M.zeros();
	A->get_diagonal(&M);
	//M.ones();
	for(int i = 0; i < A->rows(); i++)
	{
		M(i) = 1.0/M(i);
	}

	//M.ones();		// disable preconditioner
	//cout << "sparseCG_d(): preconditioner enabled" << endl;

	A->multiply(xold, &temp);		// temp := A*xold
	rold = b - temp;
	error  = rold.l2norm();		// initial residue
	if(error < tol)
	{
		cout << "sparseCG_d(): Initial residual is very small. Nothing to do." << endl;
		//x.zeros();
		return xold;
	}

	for(int i = 0; i < A->rows(); i++)
		//zold(i) = M(i)*rold(i);				// zold = M*rold
		zold(i) = rold(i);

	pold = zold;

	int steps = 0;

	do
	{
		if(steps % 10 == 0 || steps == 1)
			cout << "sparseCG_d(): Iteration " << steps << ", relative residual = " << error << endl;
		int i;

		temp1 = rold.dot_product(zold);

		A->multiply(pold, &temp);
		//temp.mprint();

		temp2 = pold.dot_product(temp);
		if(temp2 <= 0) cout << "sparseCG_d: ! Matrix A is not positive-definite!! temp2 is " << temp2 << "\n";
		theta = temp1/temp2;

		//#pragma omp parallel for default(none) private(i) shared(x,r,xold,rold,pold,temp,theta) //num_threads(nthreads_linalg)
		for(i = 0; i < x.rows(); i++)
		{
			//cout << "Number of threads " << omp_get_num_threads();
			x(i) = xold.get(i) + pold.get(i)*theta;
			rold(i) = rold.get(i) - temp.get(i)*theta;
			//diff(i) = x(i) - xold(i);
		}
		//cout << "x:\n"; x.mprint();
		//cout << "r:\n"; r.mprint();

		if(steps > 2)
		{
			//#pragma omp parallel for default(none) private(i) shared(zold,M,rold,A)
			for(i = 0; i < A->rows(); i++)
				zold(i) = M(i)*rold(i);
		}
		else
		{
			//#pragma omp parallel for default(none) private(i) shared(zold,M,rold,A)
			for(i = 0; i < A->rows(); i++)
				zold(i) = rold(i);
		}

		beta = rold.dot_product(zold) / temp1;

		//#pragma omp parallel for default(none) private(i) shared(zold,x,p,pold,beta)
		for(i = 0; i < x.rows(); i++)
			pold(i) = zold.get(i) + pold.get(i)*beta;

		//calculate ||b - A*x||
		error = rold.l2norm();
		//calculate ||x - xold||
		//error = diff.l2norm();
		//if(steps == 0) initres = error;

		// set old variables
		xold = x;
		/*rold = r;
		zold = z;
		pold = p;*/

		if(steps > maxiter)
		{
			cout << "! sparseCG_d(): Max iterations reached!\n";
			break;
		}
		steps++;
	} while(error/normalizer > tol);

	cout << "sparseCG_d(): Done. Number of iterations: " << steps << "; final residual " << error << ".\n";
	return x;
}

void precon_jacobi(SpMatrix* A, const Matrix<double>& r, Matrix<double>& z)
// Multiplies r by the Jacobi preconditioner matrix of A, and stores the result in z
{
	Matrix<double> diag(A->rows(), 1);
	A->get_diagonal(&diag);
	
	for(int i = 0; i < A->rows(); i++)
		diag(i) = 1.0/diag(i);
	
	for(int i = 0; i < A->rows(); i++)
		z(i) = diag(i)*r.get(i);
}

void precon_lusgs(SpMatrix* A, const Matrix<double>& r, Matrix<double>& z)
// Multiplies r by the LU-SGS preconditioner matrix of A, and stores the result in z
{
	SpMatrix L;
	SpMatrix U;
	Matrix<double> D(A->rows(), 1);
	Matrix<double> Dinv(A->rows(), 1);
	Matrix<double> z_initial(z.rows(), 1);
	for(int i = 0; i < z.rows(); i++)
		z_initial(i) = z.get(i);
	
	A->get_diagonal(&D);
	for(int i = 0; i < A->rows(); i++)
		Dinv(i) = 1.0/D(i);
	
	A->get_lower_triangle(L);
	A->get_upper_triangle(U);
	
	double temp = 0;
	Matrix<double> zold(A->rows(),1);
	
	// solve (D+L)*zold = r by forward substitution
	
}

Matrix<double> sparsePCG(SpMatrix* A, Matrix<double> b, Matrix<double> xold, string precon, double tol, int maxiter)
/* Calculates solution of Ax=b where A is a SPD matrix in sparse format. The preconditioner is supplied by a function pointer.*/
{
	cout << "sparsePCG(): Solving " << A->rows() << "x" << A->cols() << " system by conjugate gradient method with diagonal preconditioner\n";
	
	void (*precond)(SpMatrix* lhs, const Matrix<double>& r, Matrix<double>& z);
	//const Matrix<double>& z_initial,
	
	if(precon == "jacobi") precond = &precon_jacobi;
	else if(precon == "lusgs") precond = &precon_lusgs;
	
	// check
	//if(A->rows() != b.rows() || A->rows() != xold.rows()) cout << "sparseCG_d(): ! Mismatch in number of rows!!" << endl;

	Matrix<double> x(A->rows(),1);		// solution vector
	Matrix<double> M(A->rows(), 1);		// diagonal preconditioner, or soon, inverse of preconditioner
	Matrix<double> rold(A->rows(),1);		// initial residual = b - A*xold
	Matrix<double> r(A->rows(),1);			// residual = b - A*x
	Matrix<double> z(A->rows(),1);
	Matrix<double> zold(A->rows(),1);
	Matrix<double> p(A->rows(),1);
	Matrix<double> pold(A->rows(),1);
	Matrix<double> temp(A->rows(),1);
	Matrix<double> diff(A->rows(),1);
	double temp1, temp2;
	double theta;
	double beta;
	double error = 1.0;
	double initres;

	//cout << "sparseCG_d(): Declared everything" << endl;

	//M.ones();		// disable preconditioner
	//cout << "sparseCG_d(): preconditioner enabled" << endl;

	A->multiply(xold, &temp);		// temp := A*xold
	rold = b - temp;
	error = initres = rold.l2norm();		// initial residue
	if(error < tol)
	{
		cout << "sparsePCG(): Initial residual is very small. Nothing to do." << endl;
		//x.zeros();
		return xold;
	}

	for(int i = 0; i < A->rows(); i++)
		//zold(i) = M(i)*rold(i);				// zold = M*rold
		zold(i) = rold(i);

	pold = zold;

	int steps = 0;

	do
	{
		if(steps % 10 == 0 || steps == 1)
			cout << "sparsePCG(): Iteration " << steps << ", relative residual = " << error << endl;
		int i;

		temp1 = rold.dot_product(zold);

		A->multiply(pold, &temp);
		//temp.mprint();

		temp2 = pold.dot_product(temp);
		if(temp2 <= 0) cout << "sparsePCG: ! Matrix A is not positive-definite!! temp2 is " << temp2 << "\n";
		theta = temp1/temp2;

		//#pragma omp parallel for default(none) private(i) shared(x,r,xold,rold,pold,temp,theta) //num_threads(nthreads_linalg)
		for(i = 0; i < x.rows(); i++)
		{
			//cout << "Number of threads " << omp_get_num_threads();
			x(i) = xold.get(i) + pold.get(i)*theta;
			rold(i) = rold.get(i) - temp.get(i)*theta;
			//diff(i) = x(i) - xold(i);
		}
		//cout << "x:\n"; x.mprint();
		//cout << "r:\n"; r.mprint();

		if(steps > 5)
		{
			// calculate zold as (M^-1)*rold
			precond(A, rold, zold);
		}
		else
		{
			//#pragma omp parallel for default(none) private(i) shared(zold,M,rold,A)
			for(i = 0; i < A->rows(); i++)
				zold(i) = rold(i);
		}

		beta = rold.dot_product(zold) / temp1;

		//#pragma omp parallel for default(none) private(i) shared(zold,x,p,pold,beta)
		for(i = 0; i < x.rows(); i++)
			pold(i) = zold.get(i) + pold.get(i)*beta;

		//calculate ||b - A*x||
		error = rold.l2norm();
		//calculate ||x - xold||
		//error = diff.l2norm();
		//if(steps == 0) initres = error;

		// set old variables
		xold = x;
		/*rold = r;
		zold = z;
		pold = p;*/

		if(steps > maxiter)
		{
			cout << "! sparsePCG(): Max iterations reached!\n";
			break;
		}
		steps++;
	} while(error > tol);

	cout << "sparsePCG(): Done. Number of iterations: " << steps << "; final residual " << error << ".\n";
	return xold;
}

Matrix<double> sparse_bicgstab(SpMatrix* A, Matrix<double> b, Matrix<double> xold, double tol, int maxiter)
// Solves general linear system Ax=b using stabilized biconjugate gradient method of van der Vorst
{
	Matrix<double> rold(b.rows(), 1);
	Matrix<double> r(b.rows(), 1);
	Matrix<double> rhat(b.rows(), 1);
	Matrix<double> t(b.rows(), 1);
	Matrix<double> vold(b.rows(), 1);
	Matrix<double> v(b.rows(), 1);
	Matrix<double> pold(b.rows(), 1);
	Matrix<double> p(b.rows(), 1);
	Matrix<double> s(b.rows(), 1);
	Matrix<double> errdiff(b.rows(), 1);
	Matrix<double> x(b.rows(), 1);
	double rhoold, rho, wold, w, alpha, beta;
	double error, initres;

	A->multiply(xold, &t);		// t = A*xold, initially (t is only a temp variable here)
	rold = b - t;
	initres = error = rold.l2norm();
	int steps = 0;
	int i;

	vold.zeros(); pold.zeros();

	rhat = rold;

	rhoold = alpha = wold = 1.0;

	while(error > tol && steps <= maxiter)
	{
		if(steps % 10 == 0 || steps == 1)
			cout << "sparse_bicgstab(): Iteration " << steps << ": relative error = " << error << endl;

		rho = rhat.dot_product(rold);
		beta = rho*alpha/(rhoold*wold);

		for(i = 0; i < b.rows(); i++)
			p(i) = rold.get(i) + beta * (pold.get(i) - wold*vold.get(i));

		A->multiply(p, &v);			// v = A*p
		alpha = rho / rhat.dot_product(v);

		for(i = 0; i < b.rows(); i++)
			s(i) = rold.get(i) - alpha*v.get(i);

		if(s.dabsmax() < tol*tol)
		{
			x = xold + p*alpha;
			break;
		}

		A->multiply(s, &t);			// t = A*s

		w = t.dot_product(s)/t.dot_product(t);

		for(i = 0; i < b.rows(); i++)
		{
			x(i) = xold.get(i) + alpha*p.get(i) + w*s.get(i);
			error += (x.get(i) - xold.get(i)) * (x.get(i)-xold.get(i));
		}

		error = sqrt(error);

		for(i = 0; i < b.rows(); i++)
			r(i) = s.get(i) - w*t.get(i);

		for(i = 0; i < b.rows(); i++)
		{
			xold(i) = x.get(i);
			rold(i) = r.get(i);
			vold(i) = v.get(i);
			pold(i) = p.get(i);
		}
		rhoold = rho;
		wold = w;
		steps++;
	}

	cout << "sparse_bicgstab(): Done. Iterations: " << steps << ", final relative error: " << error << endl;

	return x;
}


}

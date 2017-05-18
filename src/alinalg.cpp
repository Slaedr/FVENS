#include "alinalg.hpp"
#include <Eigen/LU>

#define THREAD_CHUNK_SIZE 100

namespace acfd {

void gausselim(amat::Array2d<a_real>& A, amat::Array2d<a_real>& b, amat::Array2d<a_real>& x)
{
#ifdef DEBUG
	//std::cout << "gausselim: Input LHS matrix is " << A.rows() << " x " << A.cols() << std::endl;
	if(A.rows() != b.rows()) { std::cout << "gausselim: Invalid dimensions of A and b!\n"; return; }
#endif
	int N = A.rows();
	
	int k, l;
	a_real ff, temp;

	for(int i = 0; i < N-1; i++)
	{
		a_real max = fabs(A(i,i));
		int maxr = i;
		for(int j = i+1; j < N; j++)
		{
			if(fabs(A(j,i)) > max)
			{
				max = fabs(A(j,i));
				maxr = j;
			}
		}
		if(max > ZERO_TOL)
		{
			//interchange rows i and maxr 
			for(k = i; k < N; k++)
			{
				temp = A(i,k);
				A(i,k) = A(maxr,k);
				A(maxr,k) = temp;
			}
			// do the interchange for b as well
			for(k = 0; k < b.cols(); k++)
			{
				temp = b(i,k);
				b(i,k) = b(maxr,k);
				b(maxr,k) = temp;
			}
		}
		else { std::cout << "! gausselim: Pivot not found!!\n"; return; }

		for(int j = i+1; j < N; j++)
		{
			ff = A(j,i);
			for(l = i; l < N; l++)
				A(j,l) = A(j,l) - ff/A(i,i)*A(i,l);
			for(k = 0; k < b.cols(); k++)
				b(j,k) = b(j,k) - ff/A(i,i)*b(i,k);
		}
	}
	//Thus, A has been transformed to an upper triangular matrix, b has been transformed accordingly.

	//Part 2: back substitution to obtain final solution
	// Note: the solution is stored in x
	a_real sum;
	for(l = 0; l < b.cols(); l++)
	{
		x(N-1,l) = b(N-1,l)/A(N-1,N-1);

		for(int i = N-2; i >= 0; i--)
		{
			sum = 0;
			k = i+1;
			do
			{	
				sum += A(i,k)*x(k,l);
				k++;
			} while(k <= N-1);
			x(i,l) = (b(i,l) - sum)/A(i,i);
		}
	}
}

void LUfactor(amat::Array2d<a_real>& A, amat::Array2d<int>& p)
{
	int N = A.rows();
#ifdef DEBUG
	if(N != A.cols() || N != p.rows())
	{
		std::cout << "LUfactor: ! Dimension mismatch!" << std::endl;
		return;
	}
#endif
	int l,k,i,j,maxrow;
	a_real maxentry, temp;

	// set initial permutation array
	for(k = 0; k < N; k++)
		p(k) = k;

	// start
	for(k = 0; k < N-1; k++)
	{
		maxentry = fabs(A.get(p(k),k));
		maxrow = k;
		for(i = k; i < N; i++)
			if(fabs(A.get(p(i),k)) > maxentry)
			{
				maxentry = fabs(A.get(p(i),k));
				maxrow = i;
			}

		if(maxentry < ZERO_TOL)
		{
			std::cout << "! LUfactor: Encountered zero pivot! Exiting." << std::endl;
			return;
		}

		// interchange rows k and maxrow
		temp = p(k);
		p(k) = p(maxrow);
		p(maxrow) = temp;

		for(j = k+1; j < N; j++)
		{
			A(p(j),k) = A.get(p(j),k)/A.get(p(k),k);
			for(l = k+1; l < N; l++)
				A(p(j),l) -= A(p(j),k)*A.get(p(k),l);
		}
	}
}

void LUsolve(const amat::Array2d<a_real>& A, const amat::Array2d<int>& p, const amat::Array2d<a_real>& b, amat::Array2d<a_real>& x)
{
	int N = A.rows();

	amat::Array2d<a_real> y(N,1);
	a_real sum;
	int i,j;
	
	// solve Ly = Pb for y
	y(0) = b.get(p(0));
	for(i = 1; i < N; i++)
	{
		sum=0;
		for(j = 0; j < i; j++)
			sum += A.get(p(i),j)*y.get(j);
		y(i) = b.get(p(i)) - sum;
	}

	// solve Ux = y
	x(N-1) = y(N-1)/A.get(p(N-1),N-1);
#ifdef DEBUG
	if(fabs(A.get(p(N-1),N-1)) < ZERO_TOL)
		std::cout << "LUsolve: Zero diagonal element!!" << std::endl;
#endif
	for(i = N-2; i >= 0; i--)
	{
#ifdef DEBUG
		if(fabs(A.get(p(i),i)) < ZERO_TOL)
			std::cout << "LUsolve: Zero diagonal element!!" << std::endl;
#endif
		sum = 0;
		for(j = i+1; j < N; j++)
			sum += A.get(p(i),j)*x.get(j);
		x(i) = (y.get(i) - sum)/A.get(p(i),i);
	}
}

void IterativeBlockSolver::setLHS(Matrix *const diago, const Matrix *const lower, const Matrix *const upper)
{
	L = lower;
	U = upper;
	D = diago;
#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
		D[iel] = D[iel].inverse().eval();
}

void SGS_Relaxation::solve(const Matrix& __restrict__ res, Matrix& __restrict__ du)
{
	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	// we need an extra array solely to measure convergence
	Matrix uold(m->gnelem(),NVARS);

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
	}
	bnorm = std::sqrt(bnorm);

	while(resnorm/bnorm > tol && step < maxiter)
	{
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(dynamic, THREAD_CHUNK_SIZE)
			for(int iel = 0; iel < m->gnelem(); iel++) 
			{
				uold.row(iel) = du.row(iel);
				Matrix inter = Matrix::Zero(1,NVARS);
				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							inter += du.row(nbdelem)*U[face].transpose();
						}
						else {
							// lower
							inter += du.row(nbdelem)*L[face].transpose();
						}
					}
				}
				du.row(iel) = D[iel]*(-res.row(iel) - inter).transpose();
			}

#pragma omp barrier
			
			// backward sweep
#pragma omp for schedule(dynamic, THREAD_CHUNK_SIZE)
			for(int iel = m->gnelem()-1; iel >= 0; iel--) {
				Matrix inter = Matrix::Zero(1,NVARS);
				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							inter += du.row(nbdelem)*U[face].transpose();
						}
						else {
							// lower
							inter += du.row(nbdelem)*L[face].transpose();
						}
					}
				}
				du.row(iel) = D[iel]*(-res.row(iel) - inter).transpose();
			}

#pragma omp barrier

			/** Computes the `preconditioned' residual norm \f$ x^{n+1}-x^n \f$
			 * to measure convergence.
			 */
			resnorm = 0;
#pragma omp for reduction(+:resnorm)
			for(int iel = 0; iel < m->gnelem(); iel++)
			{
				// compute norm
				resnorm += (du.row(iel) - uold.row(iel)).squaredNorm();
			}
		}
		resnorm = std::sqrt(resnorm);

		step++;
	}
}

}

#include <alinalg.hpp>

namespace amat {

void gausselim(Matrix<acfd_real>& A, Matrix<acfd_real>& b, Matrix<acfd_real>& x)
{
	//std::cout << "gausselim: Input LHS matrix is " << A.rows() << " x " << A.cols() << std::endl;
	if(A.rows() != b.rows()) { std::cout << "gausselim: Invalid dimensions of A and b!\n"; return; }
	int N = A.rows();
	
	int k, l;
	acfd_real ff, temp;

	for(int i = 0; i < N-1; i++)
	{
		acfd_real max = dabs(A(i,i));
		int maxr = i;
		for(int j = i+1; j < N; j++)
		{
			if(dabs(A(j,i)) > max)
			{
				max = dabs(A(j,i));
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
	acfd_real sum;
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

LUSGS_Solver::LUSGS_Solver(const int num_vars, const UMesh2dh* const mesh, const FluxFunction* const inviscid_flux,
		const Matrix<acfd_real>* const diagonal_blocks, const Matrix<acfd_real>* const residual, Matrix<acfd_real>* const delta_u) 
	: MatrixFreeIterativeSolver(num_vars, mesh, diagonal_blocks, residual, delta_u), invf(inviscid_flux)
{
	dutemp = new Matrix<acfd_real>();
	f1.setup(nvars,1);
	f2.setup(nvars,1);
}

LUSGS_Solver::LUSGS_Solver()
{
	delete [] dutemp;
}

LUSGS_Solver::update()
{
	dutemp.zeros();
	// forward sweep
	// first compute R - L*du
	for(iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		ielem = m->gintfac(iface,0);
		n[0] = m->ggallfa(iface,0);
		n[1] = m->ggallfa(iface,1);
		s = m->ggallfa(iface,2);

		invf->evaluate(u, ielem, n, &f1);
	}
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
	}
}

}

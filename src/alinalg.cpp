#include <alinalg.hpp>

namespace amat {

void gausselim(Matrix<acfd_real>& A, Matrix<acfd_real>& b, Matrix<acfd_real>& x)
{
#ifdef DEBUG
	//std::cout << "gausselim: Input LHS matrix is " << A.rows() << " x " << A.cols() << std::endl;
	if(A.rows() != b.rows()) { std::cout << "gausselim: Invalid dimensions of A and b!\n"; return; }
#endif
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

SSOR_Solver(const int num_vars, const UMesh2dh* const mesh, const Matrix<acfd_real>* const residual, const FluxFunction* const inviscid_flux,
		const Matrix<acfd_real>* const diagonal_blocks, const Matrix<acfd_real>* const lambda_ij, const Matrix<acfd_real>* const unk, const Matrix<acfd_real>* const elem_flux,
		const acfd_real omega)
	: MatrixFreeIterativeSolver(num_vars, mesh, residual, diagonal_blocks, lambda_ij, unk, elem_flux), w(omega)
{
	f1.setup(nvars,1);
	f2.setup(nvars,1);
	uelpdu.setup(nvars,1);
	elemres.setup(nvars,1);
}

SSOR_Solver::compute_update(Matrix<acfd_real>* const du)
{
	du->zeros();
	// forward sweep
	// first compute R - L*du
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		f1.zeros();
		for(jfa = 0; jfa < m->gnfael(ielem); jfa++)
		{
			jelem = m->gesuel(ielem,jfa);
			if(jelem > ielem) continue;

			/*for(idim = 0; idim < NDIM; idim++)
			{
				ip1[idim] = m->gcoords(m->ginpoel(ielem,jfa),idim);
				ip2[idim] = m->gcoords(m->ginpoel(ielem, (jfa+1)%m->gnnode(ielem) ),idim);
			}*/
			// get normals pointing out of jelem, and *into* ielem
			/*n[0] = -1.0*(ip2[1]-ip1[1]);
			n[1] = ip2[0]-ip1[0];
			s = sqrt(n[0]*n[0] + n[1]*n[1]);*/

			iface = m->gelemface(ielem,jfa);
			n[0] = -m->ggallfa(iface,0);
			n[1] = -m->ggallfa(iface,1);
			s = m->ggallfa(iface,2);
			lambda = lambdaij->get(iface);

			for(ivar = 0; ivar < nvars; ivar++)
			{
				uelpdu(ivar) = u->get(jelem,ivar) + du->get(jelem,ivar);
			}

			// compute F(u+du*) in store in f2
			invf->compute_flux(uelpdu,n,&f2);
			// get F(u+du*) - F(u) - lambda * du
			for(ivar = 0; ivar < nvars; ivar++)
			{
				f2(ivar) = f2(ivar) - elemflux->get(jelem,ivar) - lambda*du->get(jelem);
				f2(ivar) *= s*0.5;
				f1(ivar) += f2(ivar);
			}
		}
		for(ivar = 0; ivar < nvars; ivar++)
			elemres(ivar) = res->get(ielem,ivar) - f1.get(ivar);

		gausselim(diag[ielem], elemres, du[ielem]);
	}

	// next, compute backward sweep
}

}

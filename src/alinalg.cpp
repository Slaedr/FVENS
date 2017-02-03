#include <alinalg.hpp>

namespace acfd {

void gausselim(amat::Matrix<acfd_real>& A, amat::Matrix<acfd_real>& b, amat::Matrix<acfd_real>& x)
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
		acfd_real max = fabs(A(i,i));
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

void LUfactor(amat::Matrix<acfd_real>& A, amat::Matrix<int>& p)
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
	acfd_real maxentry, temp;

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
			std::cout << "LUfactor: ! Encountered zero pivot! Exiting." << std::endl;
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

void LUsolve(const amat::Matrix<acfd_real>& A, const amat::Matrix<int>& p, const amat::Matrix<acfd_real>& b, amat::Matrix<acfd_real>& x)
{
	int N = A.rows();

	amat::Matrix<acfd_real> y(N,1);
	acfd_real sum;
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

SSOR_MFSolver::SSOR_MFSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<acfd_real>* const residual, const FluxFunction* const inviscid_flux,
		const amat::Matrix<acfd_real>* const diagonal_blocks, const amat::Matrix<int>* const perm, const amat::Matrix<acfd_real>* const lambda_ij,  const amat::Matrix<acfd_real>* const elem_flux,
		const amat::Matrix<acfd_real>* const unk, const double omega)
	: MatrixFreeIterativeSolver(num_vars, mesh, residual, inviscid_flux, diagonal_blocks, perm, lambda_ij, elem_flux, unk), w(omega)
{
	f1.setup(nvars,1);
	f2.setup(nvars,1);
	uelpdu.setup(nvars,1);
}

void SSOR_MFSolver::compute_update(amat::Matrix<acfd_real>* const du)
{
	// forward sweep
	// f1 is used to aggregate contributions from neighboring elements
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		du[ielem].zeros();
		f1.zeros();
		for(jfa = 0; jfa < m->gnfael(ielem); jfa++)
		{
			jelem = m->gesuel(ielem,jfa);
			if(jelem > ielem) continue;

			iface = m->gelemface(ielem,jfa);
			n[0] = m->ggallfa(iface,0);
			n[1] = m->ggallfa(iface,1);
			s = m->ggallfa(iface,2);
			lambda = lambdaij->get(iface);

			for(ivar = 0; ivar < nvars; ivar++)
				uelpdu(ivar) = u->get(jelem,ivar) + du[jelem].get(ivar);

			// compute F(u+du*) and store in f2
			invf->evaluate_flux(uelpdu,n,f2);
			// get F(u+du*) - F(u) - lambda * du
			for(ivar = 0; ivar < nvars; ivar++)
			{
				f2(ivar) = -1.0*(f2(ivar) - elemflux[iface].get(1,ivar)) - lambda*du[jelem].get(ivar);
				f2(ivar) *= s*0.5;
				f1(ivar) += f2(ivar);
			}
		}
		for(ivar = 0; ivar < nvars; ivar++)
			f2(ivar) = w * ((2.0-w)*res->get(ielem,ivar) - f1.get(ivar));

		LUsolve(diag[ielem], pa[ielem], f2, du[ielem]);
	}

	// next, compute backward sweep
	for(ielem = m->gnelem()-1; ielem >= 0; ielem--)
	{
		f1.zeros();
		for(jfa = 0; jfa < m->gnfael(ielem); jfa++)
		{
			jelem = m->gesuel(ielem,jfa);
			if(jelem < ielem || jelem >= m->gnelem()) continue;

			iface = m->gelemface(ielem,jfa);
			n[0] = m->ggallfa(iface,0);
			n[1] = m->ggallfa(iface,1);
			s = m->ggallfa(iface,2);
			lambda = lambdaij->get(iface);

			for(ivar = 0; ivar < nvars; ivar++)
				uelpdu(ivar) = u->get(jelem,ivar) + du[jelem].get(ivar);

			// compute F(u+du*) in store in f2
			invf->evaluate_flux(uelpdu,n,f2);
			// get F(u+du*) - F(u) - lambda * du
			for(ivar = 0; ivar < nvars; ivar++)
			{
				f2(ivar) = f2(ivar) - elemflux[iface].get(1,ivar) - lambda*du[jelem].get(ivar);
				f2(ivar) *= s*0.5;
				f1(ivar) += f2(ivar);
			}
		}

		LUsolve(diag[ielem], pa[ielem], f1, f2);
		for(ivar = 0; ivar < nvars; ivar++)
			du[ielem](ivar) -= w*f2.get(ivar);
	}
}

BJ_MFSolver::BJ_MFSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<acfd_real>* const residual, const FluxFunction* const inviscid_flux,
		const amat::Matrix<acfd_real>* const diagonal_blocks, const amat::Matrix<int>* const perm, const amat::Matrix<acfd_real>* const lambda_ij,  const amat::Matrix<acfd_real>* const elem_flux,
		const amat::Matrix<acfd_real>* const unk, const double omega)
	: MatrixFreeIterativeSolver(num_vars, mesh, residual, inviscid_flux, diagonal_blocks, perm, lambda_ij, elem_flux, unk), w(omega)
{
	f1.setup(nvars,1);
	uelpdu.setup(nvars,1);
}

void BJ_MFSolver::compute_update(amat::Matrix<acfd_real>* const du)
{
	acfd_int ielem;
	int ivar;
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		du[ielem].zeros();

		for(ivar = 0; ivar < nvars; ivar++)
			f1(ivar) = res->get(ielem,ivar);
		LUsolve(diag[ielem], pa[ielem], f1, du[ielem]);
	}
}

SSOR_Solver::SSOR_Solver(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<acfd_real>* const residual, 
		const amat::Matrix<acfd_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<acfd_real>* const lower, const amat::Matrix<acfd_real>* const upper,
		const double omega)
	: IterativeSolver(nvars, mesh, residual), D(diag), Dpa(diagperm), L(lower), U(upper), w(omega)
{
	f1.setup(nvars,1);
	f2.setup(nvars,1);
}

void SSOR_Solver::compute_update(amat::Matrix<acfd_real>* const du)
{
	int jfa, ivar;
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		du[ielem].zeros();
		f1.zeros();
		for(jfa = 0; jfa < m->gnfael(ielem); jfa++)
		{
			jelem = m->gesuel(ielem,jfa);
			if(jelem > ielem) continue;

			iface = m->gelemface(ielem,jfa);
			
			// compute L*du
			f2.zeros();
			for(i = 0; i < nvars; i++)
			{
				for(j = 0; j < nvars; j++)
					f2(i) += L[iface].get(i,j)*du[jelem].get(j);
				f1(i) += f2.get(i);
			}
		}
		for(ivar = 0; ivar < nvars; ivar++)
			f2(ivar) = w * ((2.0-w)*res->get(ielem,ivar) - f1.get(ivar));

		LUsolve(D[ielem], Dpa[ielem], f2, du[ielem]);
	}

	// next, compute backward sweep
	for(ielem = m->gnelem()-1; ielem >= 0; ielem--)
	{
		f1.zeros();
		for(jfa = 0; jfa < m->gnfael(ielem); jfa++)
		{
			jelem = m->gesuel(ielem,jfa);
			if(jelem < ielem || jelem >= m->gnelem()) continue;

			iface = m->gelemface(ielem,jfa);

			// compute U*du
			f2.zeros();
			for(i = 0; i < nvars; i++)
			{
				for(j = 0; j < nvars; j++)
					f2(i) += U[iface].get(i,j)*du[jelem].get(j);
				f1(i) += f2.get(i);
			}
		}

		LUsolve(D[ielem], Dpa[ielem], f1, f2);
		for(ivar = 0; ivar < nvars; ivar++)
			du[ielem](ivar) -= w*f2.get(ivar);
	}
}

BJ_Solver::BJ_Solver(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<acfd_real>* const residual, 
		const amat::Matrix<acfd_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<acfd_real>* const lower, const amat::Matrix<acfd_real>* const upper,
		const double omega)
	: IterativeSolver(nvars, mesh, residual), D(diag), Dpa(diagperm), w(omega)
{
	f1.setup(nvars,1);
}

void BJ_Solver::compute_update(amat::Matrix<acfd_real>* const du)
{
	acfd_int ielem;
	int ivar;
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		du[ielem].zeros();

		for(ivar = 0; ivar < nvars; ivar++)
			f1(ivar) = res->get(ielem,ivar);
		LUsolve(D[ielem], Dpa[ielem], f1, du[ielem]);
	}
}

BJ_Relaxation::BJ_Relaxation(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<acfd_real>* const residual, 
		const amat::Matrix<acfd_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<acfd_real>* const lower, const amat::Matrix<acfd_real>* const upper,
		const double omega)
	: IterativeSolver(nvars, mesh, residual), D(diag), Dpa(diagperm), L(lower), U(upper), w(omega)
{
	f1.setup(nvars,1);
}

void BJ_Relaxation::compute_update(amat::Matrix<acfd_real>* const u)
{
	// TODO: modify this function to implement block Jacobi relaxation scheme
	acfd_int ielem;
	int ivar;
	for(ielem = 0; ielem < m->gnelem(); ielem++)
	{
		u[ielem].zeros();

		for(ivar = 0; ivar < nvars; ivar++)
			f1(ivar) = res->get(ielem,ivar);
		LUsolve(D[ielem], Dpa[ielem], f1, u[ielem]);
	}
}

}

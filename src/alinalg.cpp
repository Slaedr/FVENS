#include "alinalg.hpp"
#include <algorithm>
#include <Eigen/LU>

#define THREAD_CHUNK_SIZE 200

namespace acfd {

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

/* z <- pz + qx.
 */
template<short nvars>
inline void block_axpby(const UMesh2dh *const m, const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
		const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x);
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++) {
		z.row(iel) = p*z.row(iel) + q*x.row(iel);
	}
}

/* Computes z = p b + q Ax */
template<short nvars>
void DLU_gaxpby(const UMesh2dh *const m, 
		const a_real p, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b,
		const a_real q, const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z)
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		z.row(iel) = p*b.row(iel) + q*x.row(iel)*D[iel].transpose();

		for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
		{
			a_int face = m->gelemface(iel,ifael) - m->gnbface();
			a_int nbdelem = m->gesuel(iel,ifael);

			if(nbdelem < m->gnelem())
			{
				if(nbdelem > iel) {
					// upper
					z.row(iel) += q*x.row(nbdelem)*U[face].transpose();
				}
				else {
					// lower
					z.row(iel) += q*x.row(nbdelem)*L[face].transpose();
				}
			}
		}
	}
}

template <short nvars>
PointSGS<nvars>::PointSGS(const UMesh2dh* const mesh) : DLUPreconditioner<nvars>(mesh), thread_chunk_size{500}
{
}

template <short nvars>
int PointSGS<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
{
#ifdef DEBUG
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
#endif

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(dynamic, thread_chunk_size)
		for(a_int ivar = 0; ivar < nvars*m->gnelem(); ivar++) 
		{
			a_int iel = ivar / nvars;
			int i = ivar % nvars;
			
			uold(iel,i) = z(iel,i);
			a_real inter = 0;

			for(int j = 0; j < i; j++)
				inter += D[iel](i,j)*z(iel,j);
			for(int j = i+1; j < nvars; j++)
				inter += D[iel](i,j)*z(iel,j);

			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem < m->gnelem())
				{
					if(nbdelem > iel) {
						// upper
						for(int j = 0; j < nvars; j++)
							inter += U[face](i,j) * z(nbdelem,j);
					}
					else {
						// lower
						for(int j = 0; j < nvars; j++)
							inter += L[face](i,j) * z(nbdelem,j);
					}
				}
			}
			du(iel,i) = 1.0/D[iel](i,i) * (r(iel,i) - inter);
		}

#pragma omp barrier
		
		// backward sweep
#pragma omp for schedule(dynamic, thread_chunk_size)
		for(a_int ivar = nvars*m->gnelem()-1; ivar >= 0; ivar--)
		{
			a_int iel = ivar/nvars;
			int i = ivar%nvars;
			//std::cout << iel << " " << i << std::endl;
			
			a_real inter = 0;

			for(int j = 0; j < i; j++)
				inter += D[iel](i,j)*z(iel,j);
			for(int j = i+1; j < nvars; j++)
				inter += D[iel](i,j)*z(iel,j);

			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem < m->gnelem())
				{
					if(nbdelem > iel) {
						// upper
						for(int j = 0; j < nvars; j++)
							inter += U[face](i,j) * z(nbdelem,j);
					}
					else {
						// lower
						for(int j = 0; j < nvars; j++)
							inter += L[face](i,j) * z(nbdelem,j);
					}
				}
			}
			du(iel,i) = 1.0/D[iel](i,i) * (r(iel,i) - inter);
		}
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

template <short nvars>
BlockSGS<nvars>::BlockSGS(const UMesh2dh* const mesh) : DLUPreconditioner<nvars>(mesh), thread_chunk_size{200}
{
}

template <short nvars>
void BlockSGS<nvars>::setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	L = lower;
	U = upper;
	D = diago;
#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
		D[iel] = D[iel].inverse().eval();

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

template <short nvars>
void BlockSGS<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ r, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	for(unsigned isweep = 0; isweep < napplysweeps; isweep++)
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(int iel = 0; iel < m->gnelem(); iel++) 
			{
				uold.row(iel) = z.row(iel);
				Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							inter += z.row(nbdelem)*U[face].transpose();
						}
						else {
							// lower
							inter += z.row(nbdelem)*L[face].transpose();
						}
					}
				}
				du.row(iel) = D[iel]*(r.row(iel) - inter).transpose();
			}

#pragma omp barrier
			
			// backward sweep
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(int iel = m->gnelem()-1; iel >= 0; iel--) 
			{
				Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							inter += z.row(nbdelem)*U[face].transpose();
						}
						else {
							// lower
							inter += z.row(nbdelem)*L[face].transpose();
						}
					}
				}
				du.row(iel) = D[iel]*(r.row(iel) - inter).transpose();
			}
		}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

template <short nvars>
BILU0<nvars>::BILU0(const UMesh2dh* const mesh, const unsigned short n_buildsweeps, const unsigned short n_applysweeps) 
	: DLUPreconditioner<nvars>(mesh), nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{200}
{
	luD = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
	luL = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	luU = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	y.resize(m->gnelem(),nvars);
	start = true;
}

template <short nvars>
BILU0<nvars>::~BILU0()
{
	delete [] luD;
	delete [] luL;
	delete [] luU;
}

template <short nvars>
void BILU0<nvars>::setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	L = lower;
	U = upper;
	D = diago;

	if(start) {
		// copy L,D,U for initial factorization guesses
		for(a_int iel = 0; iel < m->gnelem(); iel++)
			luD[iel] = D[iel];
		for(a_int iface = 0; iface < m->gnaface()-m->gnbface(); iface++) {
			luL[iface] = L[iface];
			luU[iface] = U[iface];
		}

		start = false;
	}

	// BILU factorization
	for(unsigned short isweep = 0; isweep < nbuildsweeps; isweep++)	
	{
			
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			/* Get lists of faces corresponding to blocks in this block-row and
			 * sort by element index.
			 * We do all this circus as we want to carry out factorization in
			 * a `Gaussian elimination' ordering, specifically in this case,
			 * the lexicographic ordering.
			 */
			
			struct LIndex { 
				a_int face;
				a_int elem;
			};
			std::vector<LIndex> lowers; lowers.reserve(m->gnfael(iel));
			std::vector<LIndex> uppers; uppers.reserve(m->gnfael(iel));

			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem < m->gnelem())
				{
					if(nbdelem < iel) {
						LIndex lower; lower.face = face; lower.elem = nbdelem;
						lowers.push_back(lower);
					}
					else {
						LIndex upper; upper.face = face; upper.elem = nbdelem;
						uppers.push_back(upper);
					}
				}
			}
			auto comp = [](LIndex i, LIndex j) { return i.elem < j.elem; }
			std::sort(lowers.begin(),lowers.end(), comp);
			std::sort(uppers.begin(),uppers.end(), comp);

			// L_ij := A_ij - sum_{k= 1 to j-1}(L_ik U_kj) U_jj^(-1)
			for(int j = 0; j < lowers.size(); j++)
			{
				Matrix<a_real,nvars,nvars,RowMajor> sum = Matrix<a_real,nvars,nvars,RowMajor>::Zero();
				
				for(int k = 0; k < j; k++) 
				{
					// first, find U_kj and see if it is non zero
					a_int otherface = -1;
					for(int ifael = 0; ifael < m->gnfael(lowers[k].elem); ifael++)
						if(lowers[j].elem == m->gesuel(lowers[k].elem,ifael))
							otherface = m->gelemface(lowers[k].elem,ifael);

					// if it's non zero, add its contribution
					if(otherface != -1)	{
						otherface -= m->gnbface();
						sum += luL[lowers[k].face] * luU[otherface];
					}
				}

				luL[lowers[j].face] = (L[lowers[j].face] - sum) * luD[lowers[j].elem].inverse();
			}

			// D_ii := A_ii - sum_{k= 1 to i-1} L_ik U_ki
			Matrix<a_real,nvars,nvars,RowMajor> sum = Matrix<a_real,nvars,nvars,RowMajor>::Zero();
			for(int k = 0; k < lowers.size(); k++) 
			{
				sum += luL[lowers[k].face] * luU[lowers[k].face];
			}
			luD[iel] = D[iel] - sum;

			// U_ij := A_ij - sum_{k= 1 to i-1} L_ik U_kj
			for(int j = 0; j < uppers.size(); j++)
			{
				Matrix<a_real,nvars,nvars,RowMajor> sum = Matrix<a_real,nvars,nvars,RowMajor>::Zero();

				for(int k = 0; k < lowers.size(); k++)
				{
					// first, find U_kj and see if it is non zero
					a_int otherface = -1;
					for(int ifael = 0; ifael < m->gnfael(lowers[k].elem); ifael++)
						if(lowers[j].elem == m->gesuel(lowers[k].elem,ifael))
							otherface = m->gelemface(lowers[k].elem,ifael);

					// if it's non zero, add its contribution
					if(otherface != -1)	{
						otherface -= m->gnbface();
						sum += luL[lowers[k].face] * luU[otherface];
					}
				}

				luU[uppers[j].face] = U[uppers[j].face] - sum;
			}
		}
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

template <short nvars>
void BILU0<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ r, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
{
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(dynamic, thread_chunk_size)
		for(int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem < m->gnelem())
				{
					if(nbdelem < iel) {
						// lower
						inter += y.row(nbdelem)*luL[face].transpose();
					}
				}
			}
			y.row(iel) = r.row(iel) - inter;
		}

#pragma omp barrier
		
#pragma omp for schedule(dynamic, thread_chunk_size)
		for(int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				if(nbdelem < m->gnelem())
				{
					if(nbdelem > iel) {
						// upper
						inter += z.row(nbdelem)*luU[face].transpose();
					}
				}
			}
			du.row(iel) = luD[iel].inverse()*(y.row(iel) - inter).transpose();
		}

#pragma omp barrier
	}
}

template <short nvars>
IterativeBlockSolver<nvars>::IterativeBlockSolver(const UMesh2dh* const mesh, std::string precond,
			const unsigned short param1, const unsigned short param2, const double param3)
	: IterativeSolver(mesh)
{
	walltime = 0; cputime = 0;
	if(precond == "PSGS") {
		prec = new PointSGS(mesh);
		std::cout << " IterativeBlockSolver: Selected point SGS preconditioner.\n";
	}
	else if(precond == "BSGS") {
		prec = new BlockSGS(mesh);
		std::cout << " IterativeBlockSolver: Selected Block SGS preconditioner.\n";
	}
	else if(precond == "BILU0") {
		prec = new BILU0(mesh, param1, param2);
		std::cout << " IterativeBlockSolver: Selected Block ILU0 preconditioner.\n";
	}
	else {
		prec = new NoPrec(mesh);
		std::cout << " IterativeBlockSolver: No preconditioning will be applied.\n";
	}
}

template <short nvars>
IterativeBlockSolver<nvars>::~IterativeBlockSolver()
{
	std::cout << " IterativeBlockSolver: CPU time = " << cputime << ", walltime = " << walltime << std::endl;
}

template <short nvars>
void IterativeBlockSolver<nvars>::setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	L = lower;
	U = upper;
	D = diago;
}

template <short nvars>
RichardsonSolver<nvars>::RichardsonSolver(const UMesh2dh *const mesh, std::string precond, 
	const unsigned short param1, const unsigned short param2, const double param3)
	: IterativeBlockSolver<nvars>(mesh, precond, param1, param2, param3)
{ }


template <short nvars>
int RichardsonSolver<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& du)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> s(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> ddu(m->gnelem(),nvars);

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
	}
	bnorm = std::sqrt(bnorm);

	while(step < maxiter)
	{
		DLU_gaxpby<nvars>(m, -1.0, res, -1.0, D,L,U, du, s);

		resnorm = 0;
#pragma omp for reduction(+:resnorm)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			// compute norm
			resnorm += s.row(iel).squaredNorm();
		}
		resnorm = std::sqrt(resnorm);
		if(resnorm/bnorm < tol) break;

		prec->apply(s, ddu);

		block_axpby<nvars>(m, 1.0, du, 1.0, ddu);

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}


template class PointSGS<NVARS>;
template class BlockSGS<NVARS>;
template class RichardsonSolver<NVARS>;
template class PointSGS<1>;
template class BlockSGS<1>;
template class RichardsonSolver<1>;

}

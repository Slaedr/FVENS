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
inline void block_axpby(const UMesh2dh *const m, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x, 
		const a_real p, const a_real q,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z)
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++) {
		z.row(iel) = p*z.row(iel) + q*x.row(iel);
	}
}

/* Computes z = p b + q Ax */
template<short nvars>
void DLU_gaxpby(const UMesh2dh *const m, 
	const Matrix<a_real,nvars,nvars,RowMajor> *const D, const Matrix<a_real,nvars,nvars,RowMajor> *const L, const Matrix<a_real,nvars,nvars,RowMajor> *const U,
	const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b, const a_real p, const a_real q,
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
IterativeBlockSolver<nvars>::IterativeBlockSolver(const UMesh2dh* const mesh)
	: IterativeSolver(mesh)
{
	walltime = 0; cputime = 0;
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
PointSGS_Relaxation<nvars>::PointSGS_Relaxation(const UMesh2dh* const mesh) : IterativeBlockSolver<nvars>(mesh), thread_chunk_size{500}
{
}

template <short nvars>
int PointSGS_Relaxation<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du)
{
#ifdef DEBUG
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
#endif

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	// we need an extra array solely to measure convergence
	Matrix<a_real,Dynamic,Dynamic,RowMajor> uold(m->gnelem(),nvars);

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
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(a_int ivar = 0; ivar < nvars*m->gnelem(); ivar++) 
			{
				a_int iel = ivar / nvars;
				int i = ivar % nvars;
				
				uold(iel,i) = du(iel,i);
				a_real inter = 0;

				for(int j = 0; j < i; j++)
					inter += D[iel](i,j)*du(iel,j);
				for(int j = i+1; j < nvars; j++)
					inter += D[iel](i,j)*du(iel,j);

				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							for(int j = 0; j < nvars; j++)
								inter += U[face](i,j) * du(nbdelem,j);
						}
						else {
							// lower
							for(int j = 0; j < nvars; j++)
								inter += L[face](i,j) * du(nbdelem,j);
						}
					}
				}
				du(iel,i) = 1.0/D[iel](i,i) * (-res(iel,i) - inter);
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
					inter += D[iel](i,j)*du(iel,j);
				for(int j = i+1; j < nvars; j++)
					inter += D[iel](i,j)*du(iel,j);

				for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
				{
					a_int face = m->gelemface(iel,ifael) - m->gnbface();
					a_int nbdelem = m->gesuel(iel,ifael);

					if(nbdelem < m->gnelem())
					{
						if(nbdelem > iel) {
							// upper
							for(int j = 0; j < nvars; j++)
								inter += U[face](i,j) * du(nbdelem,j);
						}
						else {
							// lower
							for(int j = 0; j < nvars; j++)
								inter += L[face](i,j) * du(nbdelem,j);
						}
					}
				}
				du(iel,i) = 1.0/D[iel](i,i) * (-res(iel,i) - inter);
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

	//std::cout << "   PointSGS_Relaxation: Number of steps = " << step << ", rel res = " << resnorm/bnorm << std::endl;
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

template <short nvars>
BlockSGS_Relaxation<nvars>::BlockSGS_Relaxation(const UMesh2dh* const mesh) : IterativeBlockSolver<nvars>(mesh), thread_chunk_size{200}
{
}

template <short nvars>
void BlockSGS_Relaxation<nvars>::setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
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
int BlockSGS_Relaxation<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	// we need an extra array solely to measure convergence
	Matrix<a_real,Dynamic,Dynamic,RowMajor> uold(m->gnelem(),nvars);

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
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(int iel = 0; iel < m->gnelem(); iel++) 
			{
				uold.row(iel) = du.row(iel);
				Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(int iel = m->gnelem()-1; iel >= 0; iel--) {
				Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

template <short nvars>
ABILU<nvars>::ABILU(const UMesh2dh* const mesh, const unsigned short n_buildsweeps, const unsigned short n_applysweeps) 
	: IterativeBlockSolver<nvars>(mesh), nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{200}
{
	luD = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
	luL = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	luU = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnaface()-m->gnbface()];
	y.resize(m->gnelem(),nvars);
	start = true;
}

template <short nvars>
ABILU<nvars>::~ABILU()
{
	delete [] luD;
	delete [] luL;
	delete [] luU;
}

template <short nvars>
void ABILU<nvars>::setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
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
		// TODO: copy L,D,U for initial factorization guesses

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
int ABILU<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ r, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
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
						inter += du.row(nbdelem)*luU[face].transpose();
					}
				}
			}
			du.row(iel) = luD[iel].inverse()*(y.row(iel) - inter).transpose();
		}

#pragma omp barrier
	}
}

template <short nvars>
int ABILU<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	// we need an extra array solely to measure convergence
	Matrix<a_real,Dynamic,Dynamic,RowMajor> uold(m->gnelem(),nvars);

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
	}
	bnorm = std::sqrt(bnorm);

	while(resnorm/bnorm > tol && step < maxiter)
	{
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
		resnorm = std::sqrt(resnorm);

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

template class PointSGS_Relaxation<NVARS>;
template class BlockSGS_Relaxation<NVARS>;
template class PointSGS_Relaxation<1>;
template class BlockSGS_Relaxation<1>;

}

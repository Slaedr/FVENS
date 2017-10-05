#include "alinalg.hpp"
#include <algorithm>
#include <Eigen/LU>
#include <Eigen/QR>

#define THREAD_CHUNK_SIZE 200

namespace blasted {

template <int bs>
DLUMatrix<bs>::DLUMatrix(const acfd::UMesh2dh *const mesh, 
	const short n_buildsweeps, const short n_applysweeps)
	: LinearOperator<a_real,a_int>('d'),
	  m(mesh), D(nullptr), L(nullptr), U(nullptr), luD(nullptr), luL(nullptr), luU(nullptr),
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps},
	  thread_chunk_size{200}
{
}

template <int bs>
DLUMatrix<bs>::~DLUMatrix()
{
	delete [] D;
	delete [] U;
	delete [] L;
	if(luD)
		delete [] luD;
	if(luL)
		delete [] luL;
	if(luU)
		delete [] luU;
	D=L=U=luD=luU=luL = nullptr;
}

template<int bs>
void DLUMatrix<bs>::setStructure(const a_int n, const a_int *const v1, const a_int *const v2)
{
	D = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
	L = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
	U = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
#ifdef DEBUG
	std::cout << " DLUMatrix: Storage allocated.\n";
#endif
}

template <int bs>
void DLUMatrix<bs>::setAllZero()
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
				D[iel](i,j) = 0;
#pragma omp parallel for default(shared)
	for(a_int ifa = 0; ifa < m->gnaface()-m->gnbface(); ifa++)
		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
			{
				L[ifa](i,j) = 0;
				U[ifa](i,j) = 0;
			}
}

template <int bs>
void DLUMatrix<bs>::setDiagZero()
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
				D[iel](i,j) = 0;
}

template <int bs>
void DLUMatrix<bs>::submitBlock(const a_int starti, const a_int startj,
			const a_real *const buffer,
			const a_int lud, const a_int faceid)
{
	constexpr int bs2 = bs*bs;
	const a_int startr = starti/bs;
	if(lud == 0)
		for(int i = 0; i < bs2; i++)
			D[startr].data()[i] = buffer[i];
	else if(lud == 1)
		for(int i = 0; i < bs2; i++)
			L[faceid].data()[i] = buffer[i];
	else if(lud == 2)
		for(int i = 0; i < bs2; i++)
			U[faceid].data()[i] = buffer[i];
	else {
		std::cout << "! DLUMatrix: submitBlock: Error in face index!!\n";
	}
}

template <int bs>
void DLUMatrix<bs>::updateBlock(const a_int starti, const a_int startj,
			const a_real *const buffer,
			const a_int lud, const a_int faceid)
{
	constexpr int bs2 = bs*bs;
	const a_int startr = starti/bs;
	if(lud == 0)
		for(int i = 0; i < bs2; i++)
#pragma omp atomic update
			D[startr].data()[i] += buffer[i];
	else if(lud == 1)
		for(int i = 0; i < bs2; i++)
#pragma omp atomic update
			L[faceid].data()[i] += buffer[i];
	else if(lud == 2)
		for(int i = 0; i < bs2; i++)
#pragma omp atomic update
			U[faceid].data()[i] += buffer[i];
	else {
		std::cout << "! DLUMatrix: submitBlock: Error in face index!!\n";
	}
}

template <int bs>
void DLUMatrix<bs>::updateDiagBlock(const a_int starti, const a_real *const buffer, const a_int dum)
{
	constexpr int bs2 = bs*bs;
	const a_int startr = starti/bs;
	for(int i = 0; i < bs2; i++)
#pragma omp atomic update
		D[startr].data()[i] += buffer[i];
}

template <int bs>
void DLUMatrix<bs>::apply(const a_real q, const a_real *const xx, 
                                             a_real *const __restrict zz) const
{
	Eigen::Map<const MVector> x(xx, m->gnelem(),bs);
	Eigen::Map<MVector> z(zz, m->gnelem(),bs);

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		z.row(iel).noalias() = q*x.row(iel)*D[iel].transpose();

		for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
		{
			a_int face = m->gelemface(iel,ifael) - m->gnbface();
			a_int nbdelem = m->gesuel(iel,ifael);

			if(nbdelem < m->gnelem())
			{
				if(nbdelem > iel) {
					// upper
					z.row(iel).noalias() += q*x.row(nbdelem)*U[face].transpose();
				}
				else {
					// lower
					z.row(iel).noalias() += q*x.row(nbdelem)*L[face].transpose();
				}
			}
		}
	}
}


template <int bs>
void DLUMatrix<bs>::gemv3(const a_real a, const a_real *const __restrict xx, const a_real b, 
		const a_real *const yy,
		a_real *const zz) const
{
	Eigen::Map<const MVector> x(xx, m->gnelem(),bs);
	Eigen::Map<const MVector> y(yy, m->gnelem(),bs);
	Eigen::Map<MVector> z(zz, m->gnelem(),bs);

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		z.row(iel).noalias() = b*y.row(iel) + a*x.row(iel)*D[iel].transpose();

		for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
		{
			a_int face = m->gelemface(iel,ifael) - m->gnbface();
			a_int nbdelem = m->gesuel(iel,ifael);

			if(nbdelem < m->gnelem())
			{
				if(nbdelem > iel) {
					// upper
					z.row(iel).noalias() += a*x.row(nbdelem)*U[face].transpose();
				}
				else {
					// lower
					z.row(iel).noalias() += a*x.row(nbdelem)*L[face].transpose();
				}
			}
		}
	}

/* We can also express the computation without if statement as follows.
 * We first loop over cells to compute contributions from diagonal blocks,
 * and then loop over faces, to *scatter* contributions from L and U blocks.
 *
 * This version is no better than the one above. To try,
 * uncomment this block and comment the inner loop above.
 */
#if 0
#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		a_int relem = m->gintfac(iface,1);
		a_int face = iface - m->gnbface();
		z.row(lelem).noalias() += a*x.row(relem)*U[face].transpose();
		z.row(relem).noalias() += a*x.row(lelem)*L[face].transpose();
	}
#endif
}

template <int bs>
void DLUMatrix<bs>::precJacobiSetup()
{
	if(!luD) {
		luD = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
		std::cout << " DLUMatrix: allocating lu D\n";
	}

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
		luD[iel] = D[iel].inverse();
}

template <int bs>
void DLUMatrix<bs>::precJacobiApply(const a_real *const rr, a_real *const __restrict zz) const
{
	Eigen::Map<const MVector> r(rr, m->gnelem(),bs);
	Eigen::Map<MVector> z(zz, m->gnelem(),bs);

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++) 
	{
		z.row(iel).noalias() = luD[iel] * r.row(iel).transpose();
	}
}

template <int bs>
void DLUMatrix<bs>::allocTempVector()
{
	y = MVector::Zero(m->gnelem(),bs);
}

/** \warning allocTempVector() must have been called prior to calling this method.
 */
template <int bs>
void DLUMatrix<bs>::precSGSApply(const a_real *const rr, a_real *const __restrict zz) const
{
	Eigen::Map<const MVector> r(rr, m->gnelem(),bs);
	Eigen::Map<MVector> z(zz, m->gnelem(),bs);

	// forward sweep (D+L)y = r
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,bs> inter = Matrix<a_real,1,bs>::Zero();
			for(a_int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// lower
				if(nbdelem < iel)
					inter += y.row(nbdelem)*L[face].transpose();
			}
			y.row(iel) = luD[iel]*(r.row(iel) - inter).transpose();
		}
	}

	// backward sweep (D+U)z = Dy
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,bs> inter = Matrix<a_real,1,bs>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// upper
				if(nbdelem > iel && nbdelem < m->gnelem())
					inter += z.row(nbdelem)*U[face].transpose();
			}
			z.row(iel).noalias() = luD[iel]*(y.row(iel)*D[iel].transpose() - inter).transpose();
		}
	}
}

template <int bs>
void DLUMatrix<bs>::precILUSetup()
{
	if(!luD)
	{
		luD = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
		for(a_int iel = 0; iel < m->gnelem(); iel++)
			luD[iel] = D[iel];
		std::cout << " DLUMatrix: allocating lu D\n";
	}
	if(!luL)
	{
		luL = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
		for(a_int iface = 0; iface < m->gnaface()-m->gnbface(); iface++)
			luL[iface] = L[iface];
		std::cout << " DLUMatrix: allocating lu L\n";
	}
	if(!luU)
	{
		luU = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
		for(a_int iface = 0; iface < m->gnaface()-m->gnbface(); iface++)
			luU[iface] = U[iface];
		std::cout << " DLUMatrix: allocating lu U\n";
	}
	if(y.size() <= 0)
		y = MVector::Zero(m->gnelem(),bs);

	// BILU factorization
	for(short isweep = 0; isweep < nbuildsweeps; isweep++)	
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			/* Get lists of faces corresponding to blocks in this block-row and
			 * sort by element index.
			 * We do all this circus as we want to carry out factorization in
			 * a `Gaussian elimination' ordering; specifically in this case,
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
			auto comp = [](LIndex i, LIndex j) { return i.elem < j.elem; };
			std::sort(lowers.begin(),lowers.end(), comp);
			std::sort(uppers.begin(),uppers.end(), comp);

			// L_ij := A_ij - sum_{k= 1 to j-1}(L_ik U_kj) U_jj^(-1)
			for(size_t j = 0; j < lowers.size(); j++)
			{
				Matrix<a_real,bs,bs,RowMajor> sum=Matrix<a_real,bs,bs,RowMajor>::Zero();
				
				for(size_t k = 0; k < j; k++) 
				{
					// first, find U_kj and see if it is non zero
					a_int otherface = -1;
					for(int ifael = 0; ifael < m->gnfael(lowers[k].elem); ifael++)
						if(lowers[j].elem == m->gesuel(lowers[k].elem,ifael)) {
							otherface = m->gelemface(lowers[k].elem,ifael);
							break;
						}

					// if it's non zero, add its contribution
					if(otherface != -1)	{
						otherface -= m->gnbface();
						sum += luL[lowers[k].face] * luU[otherface];
					}
				}

				luL[lowers[j].face] = (L[lowers[j].face] - sum) * luD[lowers[j].elem].inverse();
			}

			// D_ii := A_ii - sum_{k= 1 to i-1} L_ik U_ki
			Matrix<a_real,bs,bs,RowMajor> sum = Matrix<a_real,bs,bs,RowMajor>::Zero();
			for(size_t k = 0; k < lowers.size(); k++) 
			{
				sum += luL[lowers[k].face] * luU[lowers[k].face];
			}
			luD[iel] = D[iel] - sum;

			// U_ij := A_ij - sum_{k= 1 to i-1} L_ik U_kj
			for(size_t j = 0; j < uppers.size(); j++)
			{
				Matrix<a_real,bs,bs,RowMajor> sum=Matrix<a_real,bs,bs,RowMajor>::Zero();

				for(size_t k = 0; k < lowers.size(); k++)
				{
					// first, find U_kj
					a_int otherface = -1;
					for(int ifael = 0; ifael < m->gnfael(lowers[k].elem); ifael++)
						if(uppers[j].elem == m->gesuel(lowers[k].elem,ifael)) {
							otherface = m->gelemface(lowers[k].elem,ifael);
							break;
						}

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
}

template <int bs>
void DLUMatrix<bs>::precILUApply(const a_real *const rr, a_real *const __restrict zz) const
{
	Eigen::Map<const MVector> r(rr, m->gnelem(),bs);
	Eigen::Map<MVector> z(zz, m->gnelem(),bs);
	
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,bs> inter = Matrix<a_real,1,bs>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// lower
				if(nbdelem < iel)
					inter += y.row(nbdelem)*luL[face].transpose();
			}
			y.row(iel) = r.row(iel) - inter;
		}
	}

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{	
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,bs> inter = Matrix<a_real,1,bs>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// upper
				if(nbdelem > iel && nbdelem < m->gnelem()) {
					inter += z.row(nbdelem)*luU[face].transpose();
				}
			}
			z.row(iel) = luD[iel].inverse()*(y.row(iel) - inter).transpose();
		}
	}
}

template <int bs>
void DLUMatrix<bs>::printDiagnostic(const char choice) const
{
	if(choice == 'd')
		for(a_int i = 0; i < m->gnelem(); i++)
			std::cout << D[i] << std::endl << std::endl;
	else if(choice == 'l')
		for(a_int i = 0; i < m->gnaface()-m->gnbface(); i++)
			std::cout << L[i] << std::endl << std::endl;
	else if(choice == 'u')
		for(a_int i = 0; i < m->gnaface()-m->gnbface(); i++)
			std::cout << U[i] << std::endl << std::endl;
	else
		std::cout << "! DLUMatrix: printDiagnostics: Invalid choice!\n";
}

template class DLUMatrix<NVARS>;
template class DLUMatrix<1>;

} // end blasted namespace

namespace acfd {

IterativeSolverBase::IterativeSolverBase(const UMesh2dh *const mesh)
	: LinearSolver(mesh)
{
	walltime = 0; cputime = 0;
}

template <short nvars>
IterativeSolver<nvars>::IterativeSolver(const UMesh2dh* const mesh, 
		LinearOperator<a_real,a_int>* const mat, 
		Preconditioner<nvars> *const precond)
	: IterativeSolverBase(mesh), A(mat), prec(precond)
{ }

template <short nvars>
void IterativeSolver<nvars>::setupPreconditioner()
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	prec->compute();
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

// Richardson iteration
template <short nvars>
RichardsonSolver<nvars>::RichardsonSolver(const UMesh2dh *const mesh, 
		LinearOperator<a_real,a_int> *const mat,
		Preconditioner<nvars> *const precond)
	: IterativeSolver<nvars>(mesh, mat, precond)
{ }

template <short nvars>
int RichardsonSolver<nvars>::solve(const MVector& res, 
		MVector& __restrict du) const
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	const a_int N = m->gnelem()*nvars;
	MVector s(m->gnelem(),nvars);
	MVector ddu(m->gnelem(),nvars);

	// norm of RHS
	bnorm = std::sqrt(dot(N, res.data(),res.data()));

	while(step < maxiter)
	{
		A->gemv3(-1.0,du.data(), -1.0,res.data(), s.data());

		resnorm = std::sqrt(dot(N, s.data(),s.data()));
		if(resnorm/bnorm < tol) break;

		prec->apply(s.data(), ddu.data());

		axpby(N, 1.0, du.data(), 1.0, ddu.data());

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

template <short nvars>
BiCGSTAB<nvars>::BiCGSTAB(const UMesh2dh *const mesh, 
		LinearOperator<a_real,a_int> *const mat,
		Preconditioner<nvars> *const precond)
	: IterativeSolver<nvars>(mesh, mat, precond)
{ }

template <short nvars>
int BiCGSTAB<nvars>::solve(const MVector& res, 
		MVector& __restrict du) const
{
	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	const a_int N = m->gnelem()*nvars;

	a_real omega = 1.0, rho, rhoold = 1.0, alpha = 1.0, beta;
	MVector r(m->gnelem(),nvars);
	MVector rhat(m->gnelem(), nvars);
	MVector p = MVector::Zero(m->gnelem(),nvars);
	MVector v = MVector::Zero(m->gnelem(),nvars);
	MVector y(m->gnelem(),nvars);
	MVector z(m->gnelem(),nvars);
	MVector t(m->gnelem(),nvars);
	MVector g(m->gnelem(),nvars);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	// r := -res - A du
	A->gemv3(-1.0,du.data(), -1.0,res.data(), r.data());

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
		rhat.row(iel) = r.row(iel);
	}
	bnorm = std::sqrt(bnorm);

	while(step < maxiter)
	{
		// rho := rhat . r
		rho = dot(N, rhat.data(), r.data());
		beta = rho*alpha/(rhoold*omega);
		
		// p <- r + beta p - beta omega v
		axpbypcz(N, beta,p.data(), 1.0,r.data(), -beta*omega,v.data());
		
		// y <- Minv p
		prec->apply(p.data(), y.data());
		
		// v <- A y
		A->apply(1.0,y.data(), v.data());

		alpha = rho/dot(N, rhat.data(),v.data());

		// s <- r - alpha v, but reuse storage of r
		axpby(N, 1.0,r.data(), -alpha,v.data());

		// z <- Minv s
		prec->apply(r.data(), z.data());
		
		// t <- A z
		A->apply(1.0,z.data(), t.data());

		// For the more theoretically sound variant: g <- Minv t
		//prec->apply(t.data(),g.data());
		//omega = dot(N,g.data(),z.data())/dot(N,g.data(),g.data());

		omega = dot(N, t.data(),r.data()) / dot(N, t.data(),t.data());

		// du <- du + alpha y + omega z
		axpbypcz(N, 1.0,du.data(), alpha,y.data(), omega,z.data());

		// r <- r - omega t
		axpby(N, 1.0,r.data(), -omega,t.data());

		// check convergence or `lucky' breakdown
		resnorm = std::sqrt( dot(N, r.data(), r.data()) );

		//	std::cout << "   BiCGSTAB: Lin res = " << resnorm << std::endl;
		if(resnorm/bnorm < tol) break;

		rhoold = rho;
		step++;
	}

	/*if(step == maxiter)
		std::cout << " ! BiCGSTAB: Hit max iterations!\n";*/
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step+1;
}

template <short nvars>
GMRES<nvars>::GMRES(const UMesh2dh *const mesh, 
		LinearOperator<a_real,a_int> *const mat,
		Preconditioner<nvars> *const precond,
		int m_restart)
	: IterativeSolver<nvars>(mesh, mat, precond), mrestart(m_restart)
{ }

template <short nvars>
int GMRES<nvars>::solve(const MVector& res, 
		MVector& __restrict du) const
{
	const a_int N = m->gnelem()*nvars;
	int step = 0;

	MVector r(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic, ColMajor> V 
		= Matrix<a_real,Dynamic,Dynamic>::Zero(N, mrestart);
	Matrix<a_real,Dynamic,1> w = Matrix<a_real,Dynamic,1>::Zero(N);
	Matrix<a_real,Dynamic,1> y = Matrix<a_real,Dynamic,1>::Zero(N);
	Matrix<a_real,Dynamic,Dynamic> H(mrestart+1,mrestart);
	
	Matrix<a_real,Dynamic,1> be1 = Matrix<a_real,Dynamic,1>::Zero(mrestart+1);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	while(step < maxiter)
	{
		// r := -res - A du
		A->gemv3(-1.0,du.data(), -1.0,res.data(), r.data());

		prec->apply(r.data(), V.data());
		be1(0) = dot(N, V.data(),V.data());
		be1(0) = std::sqrt(be1(0));
		
#pragma omp parallel for simd default(shared)
		for(a_int k = 0; k < N; k++)
			V(k,0) = V(k,0)/be1(0);

		for(int j = 0; j < mrestart; j++)
		{
			A->apply(1.0, &V(0,j), &y(0));
			prec->apply(y.data(), w.data());

			for(int i = 0; i <= j; i++)
			{
				H(i,j) = dot(N, w.data(), &V(0,i));
				// w_j := w_j - H_(i,j) v_i
				axpby(N, 1.0,w.data(), -H(i,j),&V(0,i));
			}
			H(j+1,j) = std::sqrt( dot(N, w.data(),w.data()) );
			
			if(j < mrestart-1)
#pragma omp parallel for simd default(shared)
				for(a_int k=0; k < N; k++)
					V(k,j+1) = w(k)/H(j+1,j);
		}

		Matrix<a_real,Dynamic,1> z(mrestart);

		// Solve least-squares to get z
		Eigen::HouseholderQR<Matrix<a_real,Dynamic,Dynamic>> qr(mrestart+1,mrestart);
		qr.compute(H);
		z = qr.solve(be1);
		//z = H.householderQr().solve();

#pragma omp parallel default(shared)
		for(int i = 0; i < mrestart; i++)
		{
#pragma omp for simd
			for(a_int k = 0; k < N; k++)
				du.data()[k] += z(i) * V(k,i);
		}

		step++;
		a_real lsres = (be1 - H*z).norm()/be1(0);
		if(lsres < tol)
			break;
	}

	if(step == maxiter)
		std::cout << " ! GMRES: Hit max iterations!\n";
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step+1;
}

template <short nvars>
MFIterativeSolver<nvars>::MFIterativeSolver(const UMesh2dh* const mesh, 
		Preconditioner<nvars> *const precond, Spatial<nvars> *const spatial)
	: IterativeSolverBase(mesh), prec(precond), space(spatial)
{ }

/*template <short nvars>
MFIterativeSolver<nvars>::~MFIterativeSolver()
{
}*/

template <short nvars>
void MFIterativeSolver<nvars>::setupPreconditioner()
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	prec->compute();
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

// Richardson iteration
template <short nvars>
MFRichardsonSolver<nvars>::MFRichardsonSolver(const UMesh2dh *const mesh, 
		Preconditioner<nvars> *const precond, Spatial<nvars> *const spatial)
	: MFIterativeSolver<nvars>(mesh, precond, spatial)
{ }

template <short nvars>
int MFRichardsonSolver<nvars>::solve(const MVector& __restrict__ u,
		const amat::Array2d<a_real>& dtm,
		const MVector& __restrict__ res, 
		MVector& __restrict__ aux,
		MVector& __restrict__ du) const
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;

	// linear system residual
	MVector s(m->gnelem(), nvars);

	// linear system defect
	MVector ddu(m->gnelem(), nvars);
		//= MVector::Zero(m->gnelem(), nvars);
	// dummy
	amat::Array2d<a_real> dum;

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
	}
	bnorm = std::sqrt(bnorm);

	while(step < maxiter)
	{
#pragma omp parallel for simd default(shared)
		for(a_int i = 0; i < m->gnelem()*nvars; i++)
			(&s(0,0))[i] = 0.0;

		// compute -ve of dir derivative in the direction du, add -ve of residual, and store in s
		space->compute_jac_gemv(-1.0,res,u, du, true, dtm, -1.0,res, aux, s);

		/* compute the linear residual as follows:
		 *  b - Ax
		 *  = -r(u) - V/dt*du - dr/du(u) du
		 *  = -V/dt*du - r(u+du)
		 */
		// aux := u+du
		/*block_axpbypcz<nvars>(m, 0.0, aux, 1.0, u, 1.0, du);
		// s := r(u+du)
		space->compute_residual(aux, s, false, dum);*/

		resnorm = 0;
#pragma omp parallel for default(shared) reduction(+:resnorm)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			// s := -V/dt du - r(u+du)
			//s.row(iel) = -s.row(iel) - m->garea(iel)/dtm(iel)*du.row(iel);

			// compute norm
			resnorm += s.row(iel).squaredNorm();
		}
		resnorm = std::sqrt(resnorm);
		
		std::cout << "   MFRichardsonSolver: Lin res = " << resnorm/bnorm << std::endl;
		
		if(resnorm/bnorm < tol) break;

		prec->apply(s.data(), ddu.data());

		axpby(m->gnelem()*nvars, 1.0, du.data(), 1.0, ddu.data());

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

/// Ordering function for qsort - ascending order
/*int order_ascending(const void* a, const void* b) {
	if(*(a_int*)a < *(a_int*)b)
		return -1;
	else if(*(a_int*)a == *(a_int*)b)
		return 0;
	else
		return 1;
}*/

template <short nvars>
void setupMatrixStorage(const UMesh2dh *const m, const char mattype,
	LinearOperator<a_real,a_int> *const M)
{
	// set Jacobian storage
	if(mattype == 'd')
	{
		M->setStructure(0,nullptr,nullptr);
	}
	else if(mattype == 'c') {
		// construct non-zero structure for sparse format

		a_int* colinds, * rowptr;
		a_int nnz = (m->gnelem()+2*(m->gnaface()-m->gnbface()))*nvars*nvars;
		a_int nrows = m->gnelem()*nvars;

		rowptr = new a_int[nrows+1];
		colinds = new a_int[nnz];
		for(a_int i = 0; i < nrows+1; i++)
			rowptr[i] = nvars;

		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			// each face represents an interaction between the two cells on either side
			// we add the nonzero block to browptr but in a row shifted by 1
			for(int i = 0; i < nvars; i++) {
				rowptr[m->gintfac(iface,0)*nvars+i + 1] += nvars;
				rowptr[m->gintfac(iface,1)*nvars+i + 1] += nvars;
			}
		}
		for(a_int i = 2; i < nrows+1; i++)
			rowptr[i] += rowptr[i-1];
		rowptr[0] = 0;

		if(rowptr[nrows] != nnz)
			std::cout << "! SteadyBackwardEulerSolver:Point: Row pointer computation is wrong!!\n";
#ifdef DEBUG
		std::cout << "  SteadyBackwardEulerSolver: nnz = " << rowptr[nrows] 
			<< "; should be "<< nnz << std::endl;
#endif

		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			size_t bcsize = (rowptr[(iel+1)*nvars]-rowptr[iel*nvars])/(nvars*nvars);
			std::vector<a_int> bcinds;
			bcinds.reserve(bcsize);

			bcinds.push_back(iel);
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++) {
				a_int nbdelem = m->gesuel(iel,ifael);
				if(nbdelem < m->gnelem())
					bcinds.push_back(nbdelem);
			}

			if(bcinds.size() != bcsize)
				std::cout << "! SteadyBackwardEulerSolver:Point: Row pointers are wrong!!\n";

			std::sort(bcinds.begin(),bcinds.end());

			// i th row in element iel's block-row
			for(int i = 0; i < nvars; i++)
				// j th block of the block-row: element iel and its neighbors
				for(size_t j = 0; j < bcinds.size(); j++)
					// k th column in the j th block
					for(int k = 0; k < nvars; k++)
						colinds[rowptr[iel*nvars+i] + j*nvars + k] = bcinds[j]*nvars+k;
		}

		// Setup the point matrix
		M->setStructure(nrows, colinds,rowptr);

		delete [] rowptr;
		delete [] colinds;

#if 0
		// check
		std::cout << " CSR: Checking..\n";
		blasted::BSRMatrix<a_real,a_int,1>* B 
			= reinterpret_cast<blasted::BSRMatrix<a_real,a_int,1>*>(A);
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			std::vector<int> nbd; nbd.reserve(5);
			nbd.push_back(iel);
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				int nbelem = m->gesuel(iel,ifael);
				if(nbelem < m->gnelem())
					nbd.push_back(nbelem);
			}

			for(short j = 0; j < nvars; j++)
				if(nbd.size() != (B->browptr[(iel+1)*nvars]-B->browptr[iel*nvars])/(nvars*nvars))
					std::cout << " CSR: browptr is wrong!!\n";

			std::sort(nbd.begin(),nbd.end());
			for(short j = 0; j < nvars; j++) 
			{
				for(a_int jj = B->browptr[iel*nvars+j]; jj < B->browptr[iel*nvars+j+1]; jj++)
				{
					a_int loccol = jj - B->browptr[iel*nvars+j];
					if(B->bcolind[jj] != nbd[loccol/nvars]*nvars + loccol%nvars)
						std::cout << " CSR: bcolind is wrong!!\n";
				}
			}
		}
#endif
	}
	else {
		// construct non-zero structure for block sparse format

		a_int* bcolinds, * browptr;
		a_int nnzb = m->gnelem()+2*(m->gnaface()-m->gnbface());

		browptr = new a_int[m->gnelem()+1];
		bcolinds = new a_int[nnzb];
		for(a_int i = 0; i < m->gnelem()+1; i++)
			browptr[i] = 1;

		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			// each face represents an interaction between the two cells on either side
			// we add the nonzero block, shifted by 1 position, to browptr
			browptr[m->gintfac(iface,0)+1] += 1;
			browptr[m->gintfac(iface,1)+1] += 1;
		}
		for(a_int i = 2; i < m->gnelem()+1; i++)
			browptr[i] += browptr[i-1];
		browptr[0] = 0;

		if(browptr[m->gnelem()] != nnzb)
			std::cout << "! SteadyBackwardEulerSolver: Row pointer computation is wrong!!\n";
#ifdef DEBUG
		std::cout << "  nnz = " << browptr[m->gnelem()] << "; should be "<< nnzb << std::endl;
#endif

		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			size_t csize = browptr[iel+1]-browptr[iel];
			std::vector<a_int> cinds;
			cinds.reserve(csize);

			cinds.push_back(iel);
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++) {
				a_int nbdelem = m->gesuel(iel,ifael);
				if(nbdelem < m->gnelem())
					cinds.push_back(nbdelem);
			}

			if(cinds.size() != csize)
				std::cout << "! SteadyBackwardEulerSolver: Block-column indices are wrong!!\n";

			std::sort(cinds.begin(),cinds.end());

			for(size_t i = 0; i < cinds.size(); i++)
				bcolinds[browptr[iel]+i] = cinds[i];
		}

		// Create the matrix
		M->setStructure(m->gnelem(), bcolinds, browptr);
		delete [] browptr;
		delete [] bcolinds;
	}
}

template class RichardsonSolver<NVARS>;
template class BiCGSTAB<NVARS>;
template class GMRES<NVARS>;
template class MFRichardsonSolver<NVARS>;
template class RichardsonSolver<1>;
template class BiCGSTAB<1>;
template class GMRES<1>;

template void setupMatrixStorage<NVARS>(const UMesh2dh *const m, const char mattype,
		LinearOperator<a_real,a_int> *const A);
template void setupMatrixStorage<1>(const UMesh2dh *const m, const char mattype,
		LinearOperator<a_real,a_int> *const A);

}

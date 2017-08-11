#include "alinalg.hpp"
#include <algorithm>
#include <Eigen/LU>

#define THREAD_CHUNK_SIZE 200

namespace blasted {

template <int bs>
DLUMatrix<bs>::DLUMatrix(const acfd::UMesh2dh *const mesh, 
	const short n_buildsweeps, const short n_applysweeps)
	: LinearOperator<a_real,a_int>('d'),
	  m(mesh), D(nullptr), L(nullptr), U(nullptr), luD(nullptr), luL(nullptr), luU(nullptr),
	  nbuildsweeps(n_buildsweeps), napplysweeps(n_applysweeps),
	  thread_chunk_size(200)
{
	D = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
	L = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
	U = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
#ifdef DEBUG
	std::cout << " DLUMatrix: Setting up\n";
#endif
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
			const long lud, const long faceid)
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
			const long lud, const long faceid)
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
void DLUMatrix<bs>::updateDiagBlock(const a_int starti, const a_real *const buffer, const long dum)
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

/*// This version is no better than the one above. To try,
 * // uncomment this block and comment the inner loop above.
#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		a_int relem = m->gintfac(iface,1);
		a_int face = iface - m->gnbface();
		z.row(lelem).noalias() += a*x.row(relem)*U[face].transpose();
		z.row(relem).noalias() += a*x.row(lelem)*L[face].transpose();
	}*/
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
	MVector s(m->gnelem(),nvars);
	MVector ddu(m->gnelem(),nvars);

	// norm of RHS
	bnorm = std::sqrt(dot(res,res));

	while(step < maxiter)
	{
		A->gemv3(-1.0,du.data(), -1.0,res.data(), s.data());

		resnorm = 0;
/*#pragma omp parallel for default(shared) reduction(+:resnorm)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			// compute norm
			resnorm += s.row(iel).squaredNorm();
		}*/
		resnorm = std::sqrt(dot(s,s));
		//	std::cout << "   RichardsonSolver: Lin res = " << resnorm << std::endl;
		if(resnorm/bnorm < tol) break;

		prec->apply(s.data(), ddu.data());

		axpby(1.0, du, 1.0, ddu);

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

	a_real omega = 1.0, rho, rhoold = 1.0, alpha = 1.0, beta;
	MVector r(m->gnelem(),nvars);
	MVector rhat(m->gnelem(), nvars);
	MVector p = MVector::Zero(m->gnelem(),nvars);
	MVector v = MVector::Zero(m->gnelem(),nvars);
	//MVector g(m->gnelem(),nvars);
	MVector y(m->gnelem(),nvars);
	MVector z(m->gnelem(),nvars);
	MVector t(m->gnelem(),nvars);

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
		rho = dot(rhat, r);
		beta = rho*alpha/(rhoold*omega);
		
		// p <- r + beta p - beta omega v
		axpbypcz(beta,p, 1.0,r, -beta*omega,v);
		
		// y <- Minv p
		prec->apply(p.data(), y.data());
		
		// v <- A y
		A->apply(1.0,y.data(), v.data());

		alpha = rho/dot(rhat,v);

		// s <- r - alpha v, but reuse storage of r
		axpby(1.0,r, -alpha,v);

		prec->apply(r.data(), z.data());
		
		// t <- A z
		A->apply(1.0,z.data(), t.data());

		//prec->apply(t,g);

		//omega = block_dot<nvars>(m,g,z)/block_dot<nvars>(m,g,g);
		omega = dot(t,r)/dot(t,t);

		// du <- du + alpha y + omega z
		axpbypcz(1.0,du, alpha,y, omega,z);

		axpby(1.0,r, -omega,t);

		// check convergence or `lucky' breakdown
		resnorm = 0;
#pragma omp parallel for default(shared) reduction(+:resnorm)
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			// compute norm
			resnorm += r.row(iel).squaredNorm();
		}
		resnorm = std::sqrt(resnorm);
		//	std::cout << "   BiCGSTAB: Lin res = " << resnorm << std::endl;
		if(resnorm/bnorm < tol) break;

		rhoold = rho;
		step++;
	}

	if(step == maxiter)
		std::cout << " ! BiCGSTAB: Hit max iterations!\n";
	
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

		axpby(1.0, du, 1.0, ddu);

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step;
}

template class RichardsonSolver<NVARS>;
template class BiCGSTAB<NVARS>;
template class MFRichardsonSolver<NVARS>;
template class RichardsonSolver<1>;
template class BiCGSTAB<1>;

}

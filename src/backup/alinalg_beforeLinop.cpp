#include "alinalg.hpp"
#include <algorithm>
#include <Eigen/LU>

#define THREAD_CHUNK_SIZE 200

namespace blasted {

template <size_t bs>
DLUMatrix<bs>::DLUMatrix(const acfd::UMesh2dh *const mesh, 
	const unsigned short nbuildsweeps, const unsigned int napplysweeps)
	: m(mesh), D(nullptr), L(nullptr), U(nullptr), luD(nullptr), luL(nullptr), luU(nullptr),
	  thread_chunk_size(200)
{
	D = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
	L = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
	U = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
}

template <size_t bs>
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

template <size_t bs>
void DLUMatrix<bs>::submitBlock(const index starti, const index faceid, 
	const size_t lud, const size_t bsizej, const a_real *const buffer)
{
	constexpr size_t bs2 = bs*bs;
	if(lud == 0)
		for(size_t i = 0; i < bs2; i++)
			D[starti] = buffer[i];
	else if(lud == 1)
		for(size_t i = 0; i < bs2; i++)
			L[faceid] = buffer[i];
	else if(lud == 2)
		for(size_t i = 0; i < bs2; i++)
			U[faceid] = buffer[i];
	else {
		std::cout << "! DLUMatrix: submitBlock: Error in face index!!\n";
	}
}

template <size_t bs>
void DLUMatrix<bs>::updateBlock(const index starti, const index faceid, 
	const size_t lud, const size_t bsizej, const a_real *const buffer)
{
	constexpr size_t bs2 = bs*bs;
	if(lud == 0)
		for(size_t i = 0; i < bs2; i++)
#pragma omp atomic update
			D[starti] += buffer[i];
	else if(lud == 1)
		for(size_t i = 0; i < bs2; i++)
#pragma omp atomic update
			L[faceid] += buffer[i];
	else if(lud == 2)
		for(size_t i = 0; i < bs2; i++)
#pragma omp atomic update
			U[faceid] += buffer[i];
	else {
		std::cout << "! DLUMatrix: submitBlock: Error in face index!!\n";
	}
}

template <size_t bs>
void DLUMatrix<bs>::updateDiagBlock(const index starti, const size_t bsizei, 
		const size_t bsizej, const a_real *const buffer)
{
	constexpr size_t bs2 = bs*bs;
	for(int i = 0; i < bs2; i++)
#pragma omp atomic update
		D[starti] += buffer[i];
}

template <size_t bs>
void DLUMatrix<bs>::apply(const a_real q, const a_real *const xx, 
                                             a_real *const __restrict zz) const
{
	Eigen::Map<const Matrix<a_real,Dynamic,Dynamic,RowMajor>> x(xx, m->gnelem(),bs);
	Eigen::Map<Matrix<a_real,Dynamic,Dynamic,RowMajor>> z(zz, m->gnelem(),bs);

#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		z.row(iel) = q*x.row(iel)*D[iel].transpose();

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


template <size_t bs>
void DLUMatrix<bs>::gemv3(const a_real a, const a_real *const __restrict x, const a_real b, 
		const a_real *const y,
		a_real *const z) const
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

template <size_t bs>
void DLUMatrix<bs>::precJacobiSetup()
{
	if(!luD)
		luD = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];

#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
		luD[iel] = D[iel].inverse();
}

template <size_t bs>
void DLUMatrix<bs>::precJacobiApply(const a_real *const rr, a_real *const __restrict zz) const
{
	Eigen::Map<const Matrix<a_real,Dynamic,Dynamic,RowMajor>> r(rr, m->gnelem(),bs);
	Eigen::Map<Matrix<a_real,Dynamic,Dynamic,RowMajor>> z(zz, m->gnelem(),bs);
	constexpr size_t bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++) 
	{
		z.row(iel) = luD[iel] * r.row(iel).transpose();
	}
}

template <size_t bs>
void DLUMatrix<bs>::precSGSApply(const a_real *const rr, a_real *const __restrict zz) const
{
	Eigen::Map<const Matrix<a_real,Dynamic,Dynamic,RowMajor>> r(rr, m->gnelem(),bs);
	Eigen::Map<Matrix<a_real,Dynamic,Dynamic,RowMajor>> z(zz, m->gnelem(),bs);
	constexpr size_t bs2 = bs*bs;

	// forward sweep (D+L)y = r
	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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
	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// upper
				if(nbdelem > iel && nbdelem < m->gnelem())
					inter += z.row(nbdelem)*U[face].transpose();
			}
			z.row(iel) = luD[iel]*(y.row(iel)*D[iel].transpose() - inter).transpose();
		}
	}
}

template <size_t bs>
void DLUMatrix<bs>::precILUSetup()
{
	if(!luD)
		luD = new Matrix<a_real,bs,bs,RowMajor>[m->gnelem()];
	if(!luL)
		luL = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];
	if(!luU)
		luU = new Matrix<a_real,bs,bs,RowMajor>[m->gnaface()-m->gnbface()];

}

template <size_t bs>
void DLUMatrix<bs>::precILUApply(const a_real *const r, a_real *const __restrict z) const;

}

namespace acfd {

/* z <- pz + qx.
 */
template<short nvars>
inline void block_axpby(const UMesh2dh *const m, 
	const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
	const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x)
{
	a_real *const zz = &z(0,0); const a_real *const xx = &x(0,0);
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < m->gnelem()*nvars; i++) {
		zz[i] = p*zz[i] + q*xx[i];
	}
}

/* z <- pz + qx + ry
 */
template<short nvars>
inline void block_axpbypcz(const UMesh2dh *const m, 
	const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
	const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
	const a_real r, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& y)
{
	a_real *const zz = &z(0,0); const a_real *const xx = &x(0,0); const a_real *const yy = &y(0,0);
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < m->gnelem()*nvars; i++) {
		zz[i] = p*zz[i] + q*xx[i] + r*yy[i];
	}
}

template<short nvars>
a_real block_dot(const UMesh2dh *const m, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& a, 
	const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b)
{
	a_real sum = 0;
	const a_real *const aa = &a(0,0); const a_real *const bb = &b(0,0);
#pragma omp parallel for simd default(shared) reduction(+:sum)
	for(a_int i = 0; i < m->gnelem()*nvars; i++)
		sum += aa[i]*bb[i];

	return sum;
}

/* Computes z = q Ax */
template<short nvars>
void DLU_spmv(const UMesh2dh *const m, 
		const a_real q, const Matrix<a_real,nvars,nvars,RowMajor> *const D, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const L, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const U, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z)
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		z.row(iel) = q*x.row(iel)*D[iel].transpose();

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

/* Computes z = p b + q Ax */
template<short nvars>
void DLU_gemv(const UMesh2dh *const m, 
		const a_real p, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b,
		const a_real q, const Matrix<a_real,nvars,nvars,RowMajor> *const D, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const L, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const U, 
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

// Block Jacobi
template <short nvars>
BlockJacobi<nvars>::BlockJacobi(const UMesh2dh* const mesh) : DLUPreconditioner<nvars>(mesh)
{
	luD = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
}

template <short nvars>
BlockJacobi<nvars>::~BlockJacobi() {
	delete [] luD;
}

template <short nvars>
void BlockJacobi<nvars>::setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	D = diago;
#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
		luD[iel] = D[iel].inverse();
}

template <short nvars>
void BlockJacobi<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict z)
{

#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++) 
	{
		z.row(iel) = luD[iel] * r.row(iel).transpose();
	}
}

// Block SGS
template <short nvars>
BlockSGS<nvars>::BlockSGS(const UMesh2dh* const mesh, const unsigned short n_as) 
	: DLUPreconditioner<nvars>(mesh), napplysweeps{n_as}, thread_chunk_size{200}
{
	luD = new Matrix<a_real,nvars,nvars,RowMajor>[m->gnelem()];
	y = Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(), nvars);
}

template <short nvars>
BlockSGS<nvars>::~BlockSGS() {
	delete [] luD;
}

template <short nvars>
void BlockSGS<nvars>::setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	L = lower;
	U = upper;
	D = diago;
#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
		luD[iel] = D[iel].inverse();
}

template <short nvars>
void BlockSGS<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ r, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
{
	// forward sweep (D+L)y = r
	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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
	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(a_int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
			for(int ifael = 0; ifael < m->gnfael(iel); ifael++)
			{
				a_int face = m->gelemface(iel,ifael) - m->gnbface();
				a_int nbdelem = m->gesuel(iel,ifael);

				// upper
				if(nbdelem > iel && nbdelem < m->gnelem())
					inter += z.row(nbdelem)*U[face].transpose();
			}
			z.row(iel) = luD[iel]*(y.row(iel)*D[iel].transpose() - inter).transpose();
		}
	}
}

// Block ILU0
template <short nvars>
BILU0<nvars>::BILU0(const UMesh2dh* const mesh, 
		const unsigned short n_buildsweeps, const unsigned short n_applysweeps) 
	: DLUPreconditioner<nvars>(mesh), 
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{200}
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

/** TODO: We might need some kind of scaling of the diagonal blocks.
 */
template <short nvars>
void BILU0<nvars>::setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
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
				Matrix<a_real,nvars,nvars,RowMajor> sum= Matrix<a_real,nvars,nvars,RowMajor>::Zero();
				
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
			Matrix<a_real,nvars,nvars,RowMajor> sum = Matrix<a_real,nvars,nvars,RowMajor>::Zero();
			for(size_t k = 0; k < lowers.size(); k++) 
			{
				sum += luL[lowers[k].face] * luU[lowers[k].face];
			}
			luD[iel] = D[iel] - sum;

			// U_ij := A_ij - sum_{k= 1 to i-1} L_ik U_kj
			for(size_t j = 0; j < uppers.size(); j++)
			{
				Matrix<a_real,nvars,nvars,RowMajor> sum= Matrix<a_real,nvars,nvars,RowMajor>::Zero();

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

template <short nvars>
void BILU0<nvars>::apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ r, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ z)
{
	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(int iel = 0; iel < m->gnelem(); iel++) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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

	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{	
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(int iel = m->gnelem()-1; iel >= 0; iel--) 
		{
			Matrix<a_real,1,nvars> inter = Matrix<a_real,1,nvars>::Zero();
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

template <short nvars>
IterativeBlockSolver<nvars>::IterativeBlockSolver(const UMesh2dh* const mesh, 
		DLUPreconditioner<nvars> *const precond)
	: IterativeSolver(mesh), prec(precond)
{
	walltime = 0; cputime = 0;
}

template <short nvars>
IterativeBlockSolver<nvars>::~IterativeBlockSolver()
{
}

template <short nvars>
void IterativeBlockSolver<nvars>::setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	L = lower;
	U = upper;
	D = diago;
	prec->setLHS(diago, lower, upper);
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

// Richardson iteration
template <short nvars>
RichardsonSolver<nvars>::RichardsonSolver(const UMesh2dh *const mesh, 
		DLUPreconditioner<nvars> *const precond)
	: IterativeBlockSolver<nvars>(mesh, precond)
{ }

template <short nvars>
int RichardsonSolver<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& du)
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
		DLU_gemv<nvars>(m, -1.0, res, -1.0, D,L,U, du, s);

		resnorm = 0;
#pragma omp parallel for default(shared) reduction(+:resnorm)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			// compute norm
			resnorm += s.row(iel).squaredNorm();
		}
		resnorm = std::sqrt(resnorm);
		//	std::cout << "   RichardsonSolver: Lin res = " << resnorm << std::endl;
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

template <short nvars>
BiCGSTAB<nvars>::BiCGSTAB(const UMesh2dh *const mesh, 
		DLUPreconditioner<nvars> *const precond)
	: IterativeBlockSolver<nvars>(mesh, precond)
{ }

template <short nvars>
int BiCGSTAB<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& du)
{
	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;

	a_real omega = 1.0, rho, rhoold = 1.0, alpha = 1.0, beta;
	Matrix<a_real,Dynamic,Dynamic,RowMajor> r(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> rhat(m->gnelem(), nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> p
		= Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> v
		= Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(),nvars);
	//Matrix<a_real,Dynamic,Dynamic,RowMajor> g(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> y(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> z(m->gnelem(),nvars);
	Matrix<a_real,Dynamic,Dynamic,RowMajor> t(m->gnelem(),nvars);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	// r := -res - A du
	DLU_gemv<nvars>(m, -1.0, res, -1.0, D,L,U, du, r);

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
		rhat.row(iel) = r.row(iel);
	}
	bnorm = std::sqrt(bnorm);

	while(step < maxiter)
	{
		// rho := rhat . r
		rho = block_dot<nvars>(m, rhat, r);
		beta = rho*alpha/(rhoold*omega);
		
		// p <- r + beta p - beta omega v
		block_axpbypcz<nvars>(m, beta,p, 1.0,r, -beta*omega,v);
		
		// y <- Minv p
		prec->apply(p, y);
		
		// v <- A y
		DLU_spmv<nvars>(m, 1.0, D,L,U, y, v);

		alpha = rho/block_dot<nvars>(m,rhat,v);

		// s <- r - alpha v, but reuse storage of r
		block_axpby<nvars>(m, 1.0,r, -alpha,v);

		prec->apply(r, z);
		
		// t <- A z
		DLU_spmv<nvars>(m, 1.0, D,L,U, z, t);

		//prec->apply(t,g);

		//omega = block_dot<nvars>(m,g,z)/block_dot<nvars>(m,g,g);
		omega = block_dot<nvars>(m,t,r)/block_dot<nvars>(m,t,t);

		// du <- du + alpha y + omega z
		block_axpbypcz<nvars>(m, 1.0,du, alpha,y, omega,z);

		block_axpby<nvars>(m, 1.0,r, -omega,t);

		// check convergence or `lucky' breakdown
		resnorm = 0;
#pragma omp parallel for default(shared) reduction(+:resnorm)
		for(int iel = 0; iel < m->gnelem(); iel++)
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
	
	//block_axpby<nvars>(m, 1.0,du, alpha,y);

	if(step == maxiter)
		std::cout << " ! BiCGSTAB: Hit max iterations!\n";
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step+1;
}

template <short nvars>
MFIterativeBlockSolver<nvars>::MFIterativeBlockSolver(const UMesh2dh* const mesh, 
		DLUPreconditioner<nvars> *const precond, Spatial<nvars> *const spatial)
	: IterativeSolver(mesh), prec(precond), space(spatial)
{
	walltime = 0; cputime = 0;
}

template <short nvars>
MFIterativeBlockSolver<nvars>::~MFIterativeBlockSolver()
{
}

template <short nvars>
void MFIterativeBlockSolver<nvars>::setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	L = lower;
	U = upper;
	D = diago;
	prec->setLHS(diago, lower, upper);
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

// Richardson iteration
template <short nvars>
MFRichardsonSolver<nvars>::MFRichardsonSolver(const UMesh2dh *const mesh, 
		DLUPreconditioner<nvars> *const precond, Spatial<nvars> *const spatial)
	: MFIterativeBlockSolver<nvars>(mesh, precond, spatial)
{ }

template <short nvars>
int MFRichardsonSolver<nvars>::solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u,
		const amat::Array2d<a_real>& dtm,
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ aux,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du)
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;

	// linear system residual
	Matrix<a_real,Dynamic,Dynamic,RowMajor> s(m->gnelem(), nvars);

	// linear system defect
	Matrix<a_real,Dynamic,Dynamic,RowMajor> ddu(m->gnelem(), nvars);
		//= Matrix<a_real,Dynamic,Dynamic,RowMajor>::Zero(m->gnelem(), nvars);
	// dummy
	amat::Array2d<a_real> dum;

	// norm of RHS
#pragma omp parallel for reduction(+:bnorm) default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++)
	{
		bnorm += res.row(iel).squaredNorm();
	}
	bnorm = std::sqrt(bnorm);

	while(step < maxiter)
	{
#pragma omp parallel for simd default(shared)
		for(int i = 0; i < m->gnelem()*nvars; i++)
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
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			// s := -V/dt du - r(u+du)
			//s.row(iel) = -s.row(iel) - m->garea(iel)/dtm(iel)*du.row(iel);

			// compute norm
			resnorm += s.row(iel).squaredNorm();
		}
		resnorm = std::sqrt(resnorm);
		
		std::cout << "   MFRichardsonSolver: Lin res = " << resnorm/bnorm << std::endl;
		
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



template class NoPrec<NVARS>;
template class BlockJacobi<NVARS>;
template class PointSGS<NVARS>;
template class BlockSGS<NVARS>;
template class BILU0<NVARS>;
template class RichardsonSolver<NVARS>;
template class BiCGSTAB<NVARS>;
template class MFRichardsonSolver<NVARS>;
template class NoPrec<1>;
template class BlockJacobi<1>;
template class PointSGS<1>;
template class BlockSGS<1>;
template class BILU0<1>;
template class RichardsonSolver<1>;
template class BiCGSTAB<1>;

template void block_axpby<NVARS>(const UMesh2dh *const m, 
		const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
		const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x);

template void block_axpbypcz<NVARS>(const UMesh2dh *const m, 
		const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
		const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		const a_real r, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& y);

template a_real block_dot<NVARS>(const UMesh2dh *const m, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& a, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b);
}

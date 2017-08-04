/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef __ALINALG_H

#include <linearoperator.hpp>

#ifndef __AMESH2DH_H
#include "amesh2dh.hpp"
#endif

#ifndef __ASPATIAL_H
#include "aspatial.hpp"
#endif

#define __ALINALG_H

namespace blasted {

using acfd::a_real;
using acfd::a_int;

template <size_t bs>
class DLUMatrix : public LinearOperator<a_real,a_int>
{
protected:
	/// The mesh which describes the graph of the non-zero structure
	const acfd::UMesh2dh *const m;

	/// Diagonal blocks
	Matrix<a_real,bs,bs,RowMajor>* D;
	/// `Lower' blocks
	Matrix<a_real,bs,bs,RowMajor>* L;
	/// `Upper' blocks
	Matrix<a_real,bs,bs,RowMajor>* U;
	
	/// ILU factor - diagonal blocks (also used for BJ and SGS preconditioners)
	Matrix<a_real,bs,bs,RowMajor>* luD;
	/// ILU factor - `lower' blocks
	Matrix<a_real,bs,bs,RowMajor>* luL;
	/// ILU factor - `upper blocks
	Matrix<a_real,bs,bs,RowMajor>* luU;
	
	/// Number of sweeps used to build preconditioners
	const unsigned short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const unsigned short napplyweeps;

	/// Thread chunk size for OpenMP parallelism
	const unsigned int thread_chunk_size;

	/// Temporary array for triangular solves
	Matrix<a_real,Dynamic,Dynamic,RowMajor> y;

public:
	DLUMatrix(const acfd::UMesh2dh *const mesh, 
			const unsigned short nbuildsweeps, const unsigned int napplysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();
	
	/// Insert a block of values into the L or U not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location at the same time. As such, this function is expected to be used
	 * for populating lower and upper blocks only.
	 *
	 * \param[in] starti The block-row index, ie, cell index of the block
	 * \param[in] faceid The face index of the face shared between cells i and j
	 * \param[in] lud 0, 1 or 2, depending on whether the block being added is in the
	 *   diagonal, lower or upper part of the matrix
	 * \param[in] bsizej Dummy parameter, not used
	 * \param[in] buffer The block of values to be inserted in ROW-MAJOR ordering
	 */
	void submitBlock(const index starti, const index faceid, 
			const size_t lud, const size_t bsizej, const a_real *const buffer);

	/// Update a (contiguous) block of values into the matrix
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 *
	 * \param[in] starti The block-row index, ie, cell index of the block
	 * \param[in] faceid The INTERIOR face index of the face shared between cells i and j
	 * \param[in] lud 0, 1 or 2, depending on whether the block being updated is in the
	 *   diagonal, lower or upper part of the matrix
	 * \param[in] bsizej Dummy parameter, not used
	 * \param[in] buffer The block of values to be inserted in ROW-MAJOR ordering
	 */
	void updateBlock(const index starti, const index faceid, 
			const size_t lud, const size_t bsizej, const a_real *const buffer);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe. It's also redundant, as it's no more efficient
	 * than \ref updateBlock .
	 * \param[in] starti The block-row whose diagonal block is to be updated
	 * \param[in] bsizei Dummy
	 * \param[in] bsizej Dummy
	 * \param[in] buffer The values, in ROW-MAJOR order, making up the block to be added
	 */
	void updateDiagBlock(const index starti, const size_t bsizei, const size_t bsizej, 
			const a_real *const buffer);

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	void apply(const a_real a, const a_real *const x, a_real *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	void gemv3(const a_real a, const a_real *const __restrict x, const a_real b, 
			const a_real *const y,
			a_real *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner
	void precJacobiApply(const a_real *const r, a_real *const __restrict z) const;

	/// Allocates storage for a vector \ref ytemp required for both SGS and ILU applications
	void allocTempVector();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	void precSGSApply(const a_real *const r, a_real *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const a_real *const r, a_real *const __restrict z) const;
};

}

namespace acfd {

/// Vector addition in blocks of nvars x 1
/** z <- pz + qx.
 */
template<short nvars>
void block_axpby(const UMesh2dh *const m, 
		const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
		const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x);

/** z <- pz + qx + ry
 */
template<short nvars>
void block_axpbypcz(const UMesh2dh *const m, 
		const a_real p, Matrix<a_real,Dynamic,Dynamic,RowMajor>& z, 
		const a_real q, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		const a_real r, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& y);

/// Dot product
template<short nvars>
a_real block_dot(const UMesh2dh *const m, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& a, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b);

/// Computes z = q Ax
template<short nvars>
void DLU_spmv(const UMesh2dh *const m, 
		const a_real q, const Matrix<a_real,nvars,nvars,RowMajor> *const D, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const L, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const U, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z);

/// Computes a sparse gemv when the matrix is passed in block DLU storage
/** Specifically, computes z = pb+qAx
 */
template<short nvars>
void DLU_gemv(const UMesh2dh *const m, 
		const a_real p, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b,
		const a_real q, const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper, 
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z);

/// Preconditioner, ie, performs one iteration to solve M z = r
/** Note that subclasses do not directly perform any computation but
 * delegate all computation to the relevant subclass of LinearOperator. 
 * As such, the precise preconditioning operation applied depends on 
 * which kind of matrix the LHS is stored as.
 */
template <short nvars>
class Preconditioner
{
protected:
	LinearOperator<a_real,a_int>* A;

public:
	Preconditioner(LinearOperator<a_real,a_int> *const op)
		: A(op)
	{ }
	
	virtual ~Preconditioner()
	{ }
	
	/// Computes the preconditioning matrix M
	virtual void compute() = 0;

	/// Applies the preconditioner Mz=r
	/** \param[in] r The right hand side vector stored as a 2D array 
	 * of size nelem x nvars (nelem x 4 for 2D Euler)
	 * \param [in|out] z Contains the solution in the same format as r on exit.
	 */
	virtual void apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& z) = 0;
};

/// Do-nothing preconditioner
template <short nvars>
class NoPrec : public Preconditioner<nvars>
{
public:
	NoPrec(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op)
	{ }
	
	void compute()
	{ }
	
	void apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& z)
	{ }
};

/// Jacobi preconditioner
template <short nvars>
class BlockJacobi : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	BlockJacobi(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }
	
	void compute() {
		A->precJacobiSetup();
	}

	void apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict z) {
		A->precJacobiApply(&r(0,0), &z(0,0));
	}
};

/// Symmetric Gauss-Seidel preconditioner
template <short nvars>
class BlockSGS : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	BlockSGS(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }

	/// Sets D,L,U and inverts each D
	void compute() {
		A->precJacobiSetup();
	}

	void apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict z) {
		A->precSGSApply(&r(0,0), &z(0,0));
	}
};

/// ILU0 preconditioner
template <short nvars>
class BILU0 : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	BILU0(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }

	~BILU0();

	/// Sets D,L,U and computes the ILU factorization
	void compute() {
		A->precILUSetup();
	}
	
	/// Solves Mz=r, where M is the preconditioner
	void apply(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& r, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict z) {
		A->precILUApply(&r(0,0), &z(0,0));
	}
};

/// Base class for a linear solver
class LinearSolver
{
protected:
	const UMesh2dh* const m;						///< mesh

public:
	LinearSolver(const UMesh2dh* const mesh) 
		: m(mesh)
	{ }

	virtual ~LinearSolver()
	{ }
};

/// Preconditioned iterative solver
/** The template parameter nvars is the block size we want to use.
 * In a finite volume setting, the natural choice is the number of physical variables
 * or the number of PDEs in the system.
 */
template <short nvars>
class IterativeSolver : public LinearSolver
{
protected:
	LinearOperator *const A;        ///< The LHS matrix context
	int maxiter;                    ///< Max number of iterations
	double tol;                     ///< Tolerance
	double walltime;                ///< Stores wall-clock time measurement of solver
	double cputime;                 ///< Stores CPU time measurement of the solver

	/// Preconditioner context
	Preconditioner<nvars> *const prec;

public:
	IterativeSolver(const UMesh2dh* const mesh, Preconditioner<nvars> *const precond);

	virtual ~IterativeSolver();
	
	/// Set tolerance and max iterations
	void setParams(const double toler, const int maxits) {
		maxiter = maxits; tol = toler;
	}

	/// Compute the preconditioner
	virtual void setupPreconditioner();

	/// Solves the linear system A du = -r
	/** \param[in] res The residual vector stored as a 2D array of size nelem x nvars 
	 * (nelem x 4 for 2D Euler)
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, 
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& du) = 0;

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}
};

/// A solver that just applies the preconditioner repeatedly
template <short nvars>
class RichardsonSolver : public IterativeSolver<nvars>
{
	using IterativeSolver<nvars>::m;
	using IterativeSolver<nvars>::A;
	using IterativeSolver<nvars>::maxiter;
	using IterativeSolver<nvars>::tol;
	using IterativeSolver<nvars>::walltime;
	using IterativeSolver<nvars>::cputime;
	using IterativeSolver<nvars>::prec;

public:
	RichardsonSolver(const UMesh2dh *const mesh, Preconditioner<nvars> *const precond);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& du);
};

/// H.A. Van der Vorst's stabilized biconjugate gradient solver
/** Uses left-preconditioning only.
 */
template <short nvars>
class BiCGSTAB : public IterativeSolver<nvars>
{
	using IterativeSolver<nvars>::m;
	using IterativeSolver<nvars>::A;
	using IterativeSolver<nvars>::maxiter;
	using IterativeSolver<nvars>::tol;
	using IterativeSolver<nvars>::walltime;
	using IterativeSolver<nvars>::cputime;
	using IterativeSolver<nvars>::prec;

public:
	BiCGSTAB(const UMesh2dh *const mesh, Preconditioner<nvars> *const precond);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& du);
};

/// Base class for matrix-free solvers
/** Note that subclasses are matrix-free only with regard to the top-level solver,
 * usually a Krylov subspace solver. The preconditioning matrix is still computed and stored.
 */
template <short nvars>
class MFIterativeSolver : public IterativeSolver
{
protected:
	/// (Inverted) diagonal blocks of LHS (Jacobian) matrix
	const Matrix<a_real,nvars,nvars,RowMajor>* D;
	/// `Lower' blocks of LHS
	const Matrix<a_real,nvars,nvars,RowMajor>* L;
	/// `Upper' blocks of LHS
	const Matrix<a_real,nvars,nvars,RowMajor>* U;
	/// Preconditioner context
	Preconditioner<nvars> *const prec;
	/// Spatial discretization context needed for matrix-vector product
	Spatial<nvars>* const space;
	
	double walltime;
	double cputime;

public:
	MFIterativeSolver(const UMesh2dh* const mesh, Preconditioner<nvars> *const precond,
		Spatial<nvars> *const spatial);

	virtual ~MFIterativeSolver();

	/// Sets D,L,U for preconditioner
	virtual void setLHS(const Matrix<a_real,nvars,nvars,RowMajor> *const diago, 
			const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
			const Matrix<a_real,nvars,nvars,RowMajor> *const upper);

	/// Solves the linear system A du = -r
	/** \param[in] u The state at which the Jacobian and RHS res have been computed
	 * \param[in] res The residual vector stored as a 2D array of size nelem x nvars 
	 * (nelem x 4 for 2D Euler)
	 * \param aux Temporary storage needed for matrix-free evaluation of Jacobian-vector products
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
		const amat::Array2d<a_real>& dtm,
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ aux,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du) = 0;

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}
};

/// A matrix-free solver that just applies the preconditioner repeatedly
/// in a defect-correction iteration.
template <short nvars>
class MFRichardsonSolver : public MFIterativeSolver<nvars>
{
	using MFIterativeSolver<nvars>::m;
	using MFIterativeSolver<nvars>::maxiter;
	using MFIterativeSolver<nvars>::tol;
	using MFIterativeSolver<nvars>::D;
	using MFIterativeSolver<nvars>::L;
	using MFIterativeSolver<nvars>::U;
	using MFIterativeSolver<nvars>::walltime;
	using MFIterativeSolver<nvars>::cputime;
	using MFIterativeSolver<nvars>::prec;
	using MFIterativeSolver<nvars>::space;

public:
	MFRichardsonSolver(const UMesh2dh *const mesh, Preconditioner<nvars> *const precond,
			Spatial<nvars> *const spatial);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ u, 
		const amat::Array2d<a_real>& dtm,
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ res, 
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ aux,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& __restrict__ du);
};


}
#endif

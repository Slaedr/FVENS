/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef __ALINALG_H

#ifndef __AMESH2DH_H
#include "amesh2dh.hpp"
#endif

#define __ALINALG_H

namespace acfd {

/// Solves Ax=b for dense A by Gaussian elimination
void gausselim(amat::Array2d<a_real>& A, amat::Array2d<a_real>& b, amat::Array2d<a_real>& x);

/// Factors the dense matrix A into unit lower triangular matrix and upper triangular matrix in place, with partial pivoting
void LUfactor(amat::Array2d<a_real>& A, amat::Array2d<int>& p);

/// Solve LUx = b
/** \param A contains L and U which are assumed dense
 * \param p is the permutation array
 * \param b is the RHS
 * \param x will contain the solution
 */
void LUsolve(const amat::Array2d<a_real>& A, const amat::Array2d<int>& p, const amat::Array2d<a_real>& b, amat::Array2d<a_real>& x);

/// Vector addition in blocks of nvars x 1
/** z <- pz + qx.
 */
template<short nvars>
void block_axpby(const UMesh2dh *const m, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x, const a_real p, const a_real q,
			Matrix<a_real,Dynamic,Dynamic,RowMajor>& z);

/// Computes a sparse gaxpy when the matrix is passed in block DLU storage
/** Specifically, computes z = pb+qAx
 */
template<short nvars>
void DLU_gaxpby(const UMesh2dh *const m, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
		const Matrix<a_real,nvars,nvars,RowMajor> *const upper,
		const Matrix<a_real,Dynamic,Dynamic,RowMajor>& x, const Matrix<a_real,Dynamic,Dynamic,RowMajor>& b, const a_real p, const a_real q,
		Matrix<a_real,Dynamic,Dynamic,RowMajor>& z);

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

/// Abstract class for iterative linear solvers
class IterativeSolver : public LinearSolver
{
protected:
	int maxiter;					///< Max number of iterations
	double tol;						///< Tolerance

public:
	IterativeSolver(const UMesh2dh *const mesh)
		: LinearSolver(mesh)
	{ }

	/// Set tolerance and max iterations
	void setParams(const double toler, const int maxits) {
		maxiter = maxits; tol = toler;
	}

	virtual ~IterativeSolver()
	{ }
};

/// Iterative solver in which the LHS is stored in a block D,L,U format
/** The template parameter nvars is the block size we want to use.
 * In a finite volume setting, the natural choice is the number of physical variables
 * or the number of PDEs in the system.
 */
template <short nvars>
class IterativeBlockSolver : public IterativeSolver
{
protected:
	Matrix<a_real,nvars,nvars,RowMajor> * D;					///< (Inverted) diagonal blocks of LHS (Jacobian) matrix
	const Matrix<a_real,nvars,nvars,RowMajor>* L;				///< `Lower' blocks of LHS
	const Matrix<a_real,nvars,nvars,RowMajor>* U;				///< `Upper' blocks of LHS
	double walltime;
	double cputime;

public:
	IterativeBlockSolver(const UMesh2dh* const mesh);

	/// Sets D,L,U
	virtual void setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
			const Matrix<a_real,nvars,nvars,RowMajor> *const upper);

	/// Solves the linear system
	/** \param[in] res The right hand side vector stored as a 2D array of size nelem x nvars (nelem x 4 for 2D Euler)
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& du) = 0;

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}
};

/// Full matrix storage version of the (point) symmetric Gauss-Seidel solver
template <short nvars>
class PointSGS_Relaxation : public IterativeBlockSolver<nvars>
{
	using LinearSolver::m;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeBlockSolver<nvars>::D;
	using IterativeBlockSolver<nvars>::L;
	using IterativeBlockSolver<nvars>::U;
	using IterativeBlockSolver<nvars>::walltime;
	using IterativeBlockSolver<nvars>::cputime;

	const int thread_chunk_size;

public:
	PointSGS_Relaxation(const UMesh2dh* const mesh);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& du);
};

/// Full matrix storage version of the block symmetric Gauss-Seidel solver
template <short nvars>
class BlockSGS_Relaxation : public IterativeBlockSolver<nvars>
{
	using LinearSolver::m;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeBlockSolver<nvars>::D;
	using IterativeBlockSolver<nvars>::L;
	using IterativeBlockSolver<nvars>::U;
	using IterativeBlockSolver<nvars>::walltime;
	using IterativeBlockSolver<nvars>::cputime;

	const int thread_chunk_size;

public:
	BlockSGS_Relaxation(const UMesh2dh* const mesh);

	/// Sets D,L,U and inverts each D
	void setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
			const Matrix<a_real,nvars,nvars,RowMajor> *const upper);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& du);
};

/// Asynchronous block ILU defect-correction solver
/** Since this is a defect-correction type iteration, we need to have access to the original matrix.
 * Thus we need to store the LU-factored blocks separately.
 */
template <short nvars>
class ABILU : public IterativeBlockSolver<nvars>
{
	using LinearSolver::m;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeBlockSolver<nvars>::D;
	using IterativeBlockSolver<nvars>::L;
	using IterativeBlockSolver<nvars>::U;
	using IterativeBlockSolver<nvars>::walltime;
	using IterativeBlockSolver<nvars>::cputime;
	
	Matrix<a_real,nvars,nvars,RowMajor> * luD;
	Matrix<a_real,nvars,nvars,RowMajor> * luL;
	Matrix<a_real,nvars,nvars,RowMajor> * luU;

	const int thread_chunk_size;

public:
	ABILU(const UMesh2dh* const mesh);

	/// Sets D,L,U and computes the ILU factorization
	void setLHS(Matrix<a_real,nvars,nvars,RowMajor> *const diago, const Matrix<a_real,nvars,nvars,RowMajor> *const lower, 
			const Matrix<a_real,nvars,nvars,RowMajor> *const upper);

	int solve(const Matrix<a_real,Dynamic,Dynamic,RowMajor>& res, Matrix<a_real,Dynamic,Dynamic,RowMajor>& du);
};

}
#endif

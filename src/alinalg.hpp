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

/// Base class for a linear solver
class LinearSolver
{
protected:
	const UMesh2dh* const m;						///< mesh

public:
	LinearSolver(const UMesh2dh* const mesh) 
		: m(mesh)
	{ }

	/// Solves the linear system
	/** \param[in] res The right hand side vector stored as a 2D array of size nelem x NVARS (nelem x 4 for 2D Euler)
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 */
	virtual void solve(const Matrix& res, Matrix& du) = 0;

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
//template <typename Matrixb>
class IterativeBlockSolver : public IterativeSolver
{
protected:
	Matrixb * D;						///< (Inverted) diagonal blocks of LHS (Jacobian) matrix
	const Matrixb * L;					///< `Lower' blocks of LHS
	const Matrixb * U;					///< `Upper' blocks of LHS
	double walltime;
	double cputime;

public:
	IterativeBlockSolver(const UMesh2dh* const mesh);

	/// Sets D,L,U
	virtual void setLHS(Matrixb *const diago, const Matrixb *const lower, const Matrixb *const upper);

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}

	/// Solves the linear system with D,L,U as the LHS and the argument res as the negative of the RHS, and stores the result in du
	virtual void solve(const Matrix& res, Matrix& du) = 0;
};

/// Full matrix storage version of the (point) symmetric Gauss-Seidel solver
//template <typename Matrixb>
class PointSGS_Relaxation : public IterativeBlockSolver
{
	const int thread_chunk_size;

public:
	PointSGS_Relaxation(const UMesh2dh* const mesh);

	void solve(const Matrix& res, Matrix& du);
};

/// Full matrix storage version of the block symmetric Gauss-Seidel solver
//template <typename Matrixb>
class BlockSGS_Relaxation : public IterativeBlockSolver
{
	const int thread_chunk_size;

public:
	BlockSGS_Relaxation(const UMesh2dh* const mesh);

	/// Sets D,L,U and inverts each D
	void setLHS(Matrixb *const diago, const Matrixb *const lower, const Matrixb *const upper);

	void solve(const Matrix& res, Matrix& du);
};

}
#endif

/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef __ALINALG_H

#ifndef __AMESH2DH_H
#include <amesh2dh.hpp>
#endif

#ifndef __AEULERFLUX_H
#include "aeulerflux.hpp"
#endif

#define __ALINALG_H

namespace acfd {

/// Solves Ax=b for dense A by Gaussian elimination
void gausselim(amat::Matrix<a_real>& A, amat::Matrix<a_real>& b, amat::Matrix<a_real>& x);

/// Factors the dense matrix A into unit lower triangular matrix and upper triangular matrix in place, with partial pivoting
void LUfactor(amat::Matrix<a_real>& A, amat::Matrix<int>& p);

/// Solve LUx = b
/** \param A contains L and U which are assumed dense
 * \param p is the permutation array
 * \param b is the RHS
 * \param x will contain the solution
 */
void LUsolve(const amat::Matrix<a_real>& A, const amat::Matrix<int>& p, const amat::Matrix<a_real>& b, amat::Matrix<a_real>& x);

/// Base class for a linear solver
class LinearSolver
{
protected:
	const UMesh2dh* const m;						///< mesh

public:
	LinearSolver(const UMesh2dh* const mesh) 
		: m(mesh)
	{ }

	/// Solves the linear system with D,L,U as the LHS and the argument -res as the RHS and stores the result in du
	virtual void solve(const Eigen::Matrix& res, Eigen::Matrix& du) = 0;

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
	IterativeSolver(const Umesh2dh *const mesh, const int maxiterations, const double tolerance)
		: LinearSolver(mesh), maxiter(maxiterations), tol(tolerance)
	{ }

	/// Solves the linear system with D,L,U as the LHS and the argument -res as the RHS and stores the result in du
	virtual void solve(const Eigen::Matrix& res, Eigen::Matrix& du) = 0;

	virtual ~IterativeSolver()
	{ }
};

/// Iterative solver in which the LHS is stored in a block D,L,U format
class IterativeBlockSolver : public IterativeSolver
{
protected:
	Eigen::Matrix * D;							///< (Inverted) diagonal blocks of LHS (Jacobian) matrix
	const Eigen::Matrix * L;					///< `Lower' blocks of LHS
	const Eigen::Matrix * U;					///< `Upper' blocks of LHS

public:
	IterativeBlockSolver(const UMesh2dh* const mesh, const int maxits, const double toler) 
		: IterativeSolver(mesh, maxits, toler)
	{ }

	/// Sets D,L,U and inverts each D
	void setLHS(Eigen::Matrix *const diago, const Eigen::Matrix *const lower, const Eigen::Matrix *const upper);

	/// Solves the linear system with D,L,U as the LHS and the argument res as the negative of the RHS, and stores the result in du
	virtual void solve(const Eigen::Matrix& res, Eigen::Matrix& du) = 0;

	virtual ~IterativeSolver()
	{ }
};

/// Full matrix storage version of the symmetric Gauss-Seidel solver
class SGS_Relaxation : public IterativeBlockSolver
{
public:
	SGS_Relaxation(const UMesh2dh* const mesh, const int maxits, const double toler) : IterativeBlockSolver(mesh, maxits, toler) { }

	void solve(const Eigen::Matrix& res, Eigen::Matrix& du);
};

}
#endif

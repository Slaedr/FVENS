/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef __ALINALG_H

#ifndef __AMESH2DH_H
#include <amesh2dh.hpp>
#endif

#ifndef __AEULERFLUXNORMAL_H
#include <aeulerfluxnormal.hpp>
#endif

#define __ALINALG_H

namespace acfd {

/// Solves Ax=b for dense A by Gaussian elimination
void gausselim(amat::Matrix<a_real>& A, amat::Matrix<a_real>& b, amat::Matrix<a_real>& x);

/// Factors A into unit lower triangular matrix and upper triangular matrix in place, with partial pivoting
void LUfactor(amat::Matrix<a_real>& A, amat::Matrix<int>& p);

/// Solve LUx = b
/** \param A contains L and U
 * \param p is the permutation array
 * \param b is the RHS
 * \param x will contain the solution
 */
void LUsolve(const amat::Matrix<a_real>& A, const amat::Matrix<int>& p, const amat::Matrix<a_real>& b, amat::Matrix<a_real>& x);

/// Base class for AF (approximate factorization)-type iterative linear solver
class IterativeSolver
{
protected:
	const int nvars;								///< Number of conserved variables
	const UMesh2dh* const m;						///< mesh
	const amat::Matrix<a_real>* const res;		///< Vector of `residuals' at previous iteration
public:
	IterativeSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual) 
		: nvars(num_vars), m(mesh), res(residual)
	{ }

	/// Supposed to carry a single step of the corresponding AF solver and store the correction in the argument
	virtual void compute_update(amat::Matrix<a_real>* const deltau) = 0;

	virtual ~IterativeSolver()
	{ }
};

/// Base class for matrix-free iterative solvers of AF-type
/** A FluxFunction object allows us to compute inviscid flux corresponding to a given state vector.
 * This is used to compute the action of the lower/upper parts of the (approximate) Jacobian on a given vector.
 * Adapted from reference: 
 * H. Luo, D. Sharov, J.D. Baum, R. Loehner. "On the Computation of Compressible Turbulent Flows on Unstructured Grids". Intl. Journal of Computational Fluid Dynamics Vol 14, No 4, pp 253-270. 2001.
 */
class MatrixFreeIterativeSolver : public IterativeSolver
{
protected:
	const FluxFunction* const invf;
	const amat::Matrix<a_real>* const diag;			///< Diagonal blocks
	const amat::Matrix<int>* const pa;					///< permuation array of diag after LU factorization
	const amat::Matrix<a_real>* const lambdaij;		///< Eigenvalue part of the Jacobian
	const amat::Matrix<a_real>* const elemflux;		///< Flux corresponding to the state at which Jacobian is to be computed
	const amat::Matrix<a_real>* const u;				///< state at which Jacobian is to be computed
public:
	MatrixFreeIterativeSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, const FluxFunction* const inviscid_flux,
			const amat::Matrix<a_real>* const diagonal_blocks, const amat::Matrix<int>* const perm, const amat::Matrix<a_real>* const lambda_ij, const amat::Matrix<a_real>* const elem_flux,
			const amat::Matrix<a_real>* const unk)
		: IterativeSolver(num_vars, mesh, residual), invf(inviscid_flux), diag(diagonal_blocks), pa(perm), lambdaij(lambda_ij), elemflux(elem_flux), u(unk)
	{
	}
	
	virtual void compute_update(amat::Matrix<a_real>* const deltau) = 0;
};

/// Matrix-free SSOR solver
/** Adapted from reference: 
 * H. Luo, D. Sharov, J.D. Baum and R. Loehner. "On the Computation of Compressible Turbulent Flows on Unstructured Grids". Internation Journal of Computational Fluid Dynamics Vol14, No 4, pp 253-270.
 * 2001.
 */
class SSOR_MFSolver : public MatrixFreeIterativeSolver
{
	
	amat::Matrix<a_real> f1;
	amat::Matrix<a_real> f2;
	amat::Matrix<a_real> uelpdu;
	amat::Matrix<a_real> elemres;
	const double w;								///< Over-relaxation factor
	a_int ielem, jelem, iface;
	a_real s, sum;
	a_real n[NDIM];
	int jfa;
	int ivar;
	a_real lambda;
	
public:

	SSOR_MFSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, const FluxFunction* const inviscid_flux,
			const amat::Matrix<a_real>* const diagonal_blocks, const amat::Matrix<int>* const perm, const amat::Matrix<a_real>* const lambda_ij, const amat::Matrix<a_real>* const elem_flux,
			const amat::Matrix<a_real>* const unk, const double omega);

	/// Carries out a single step (one forward followed by one backward sweep) of SSOR and stores the correction in the argument.
	/** \note NOTE: Make sure deltau is an array of length nelem of type Matrix<a_real>(nvars,1).
	 */
	void compute_update(amat::Matrix<a_real>* const deltau);
};

/// Block Jacobi solver
class BJ_MFSolver : public MatrixFreeIterativeSolver
{
	
	amat::Matrix<a_real> f1;
	amat::Matrix<a_real> f2;
	amat::Matrix<a_real> uelpdu;
	amat::Matrix<a_real> elemres;
	const double w;								///< Over-relaxation factor
	a_int ielem, jelem, iface;
	a_real s, sum;
	a_real n[NDIM];
	int jfa;
	int ivar;
	a_real lambda;
	
public:

	BJ_MFSolver(const int num_vars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, const FluxFunction* const inviscid_flux,
			const amat::Matrix<a_real>* const diagonal_blocks, const amat::Matrix<int>* const perm, const amat::Matrix<a_real>* const lambda_ij, const amat::Matrix<a_real>* const elem_flux,
			const amat::Matrix<a_real>* const unk, const double omega);

	/// Carries out a single step (one forward followed by one backward sweep) of SSOR and stores the correction in the argument.
	/** \note NOTE: Make sure deltau is an array of length nelem of type Matrix<a_real>(nvars,1).
	 */
	void compute_update(amat::Matrix<a_real>* const deltau);
};

/// Full matrix storage version of the SSOR solver
class SSOR_Solver : public IterativeSolver
{
	const amat::Matrix<a_real>* const D;
	const amat::Matrix<int>* const Dpa;
	const amat::Matrix<a_real>* const U;
	const amat::Matrix<a_real>* const L;
	const double w;
	amat::Matrix<a_real> f1;
	amat::Matrix<a_real> f2;
	a_int i, j, ielem, jelem, iface;

public:
	SSOR_Solver(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, 
			const amat::Matrix<a_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<a_real>* const lower, const amat::Matrix<a_real>* const upper,
			const double omega);

	void compute_update(amat::Matrix<a_real>* const deltau);
};

/// Full matrix storage version of the BJ solver
class BJ_Solver : public IterativeSolver
{
	const amat::Matrix<a_real>* const D;
	const amat::Matrix<int>* const Dpa;
	const double w;
	
	amat::Matrix<a_real> f1;
	amat::Matrix<a_real> f2;
	a_int ielem, jelem, iface;
	a_real s, sum;
	a_real n[NDIM];
	int jfa;
	int ivar;

public:
	BJ_Solver(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, 
			const amat::Matrix<a_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<a_real>* const lower, const amat::Matrix<a_real>* const upper,
			const double omega);

	void compute_update(amat::Matrix<a_real>* const deltau);
};

class BJ_Relaxation : public IterativeSolver
{
	const amat::Matrix<a_real>* const D;
	const amat::Matrix<int>* const Dpa;
	const amat::Matrix<a_real>* const U;
	const amat::Matrix<a_real>* const L;
	const double w;
	
	amat::Matrix<a_real> f1;
	amat::Matrix<a_real> f2;
	a_int ielem, jelem, iface;
	a_real s, sum;
	a_real n[NDIM];
	int jfa;
	int ivar;

public:
	BJ_Relaxation(const int nvars, const UMesh2dh* const mesh, const amat::Matrix<a_real>* const residual, 
			const amat::Matrix<a_real>* const diag, const amat::Matrix<int>* const diagperm, const amat::Matrix<a_real>* const lower, const amat::Matrix<a_real>* const upper,
			const double omega);

	void compute_update(amat::Matrix<a_real>* const deltau);
};

}
#endif

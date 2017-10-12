/** @file alinalg.hpp
 * @brief Linear algebra subroutines
 * @author Aditya Kashi
 */

#ifndef __ALINALG_H

#include <linearoperator.hpp>

#include "aconstants.hpp"

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

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;

using acfd::MVector;

/// A sparse matrix stored in a `DLU' format
/** Includes some BLAS 2 and preconditioning operations
 */
template <int bs>
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
	const short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const short napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const unsigned int thread_chunk_size;

	/// Temporary array for triangular solves
	mutable MVector y;

public:

	/// Sets data; does not allocate any storage \sa setStructure
	DLUMatrix(const acfd::UMesh2dh *const mesh, 
			const short nbuildsweeps, const short napplysweeps);

	/// De-allocates memory
	virtual ~DLUMatrix();

	/// Allocates storage using the mesh context; arguments can be 0, NULL and NULL
	void setStructure(const a_int n, const a_int *const v1, const a_int *const v2);

	/// Sets storage D, L and U to zero
	void setAllZero();

	/// Sets diagonal blocks D to zero
	void setDiagZero();
	
	/// Insert a block of values into the L or U not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location at the same time. As such, this function is expected to be used
	 * for populating lower and upper blocks only.
	 *
	 * \param[in] starti The row index, ie, cell index of the block times nvars
	 * \param[in] startj The column index
	 * \param[in] buffer The block of values to be inserted in ROW-MAJOR ordering
	 * \param[in] lud 0, 1 or 2, depending on whether the block being updated is in the
	 *   diagonal, lower or upper part of the matrix
	 * \param[in] faceid The INTERIOR face index of the face shared between cells i and j
	 */
	void submitBlock(const a_int starti, const a_int startj, 
			const a_real *const buffer,
			const a_int lud, const a_int face_id);

	/// Update a (contiguous) block of values into the matrix
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 *
	 * \param[in] starti The row index, ie, cell index of the block times nvars
	 * \param[in] startj The column index
	 * \param[in] buffer The block of values to be inserted in ROW-MAJOR ordering
	 * \param[in] lud 0, 1 or 2, depending on whether the block being updated is in the
	 *   diagonal, lower or upper part of the matrix
	 * \param[in] faceid The INTERIOR face index of the face shared between cells i and j
	 */
	void updateBlock(const a_int starti, const a_int startj, 
			const a_real *const buffer,
			const a_int lud, const a_int face_id);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe. It's also redundant, as it's no more efficient
	 * than \ref updateBlock .
	 * \param[in] starti The block-row whose diagonal block is to be updated
	 * \param[in] buffer The values, in ROW-MAJOR order, making up the block to be added
	 * \param[in] dummy Any integer value, not used.
	 */
	void updateDiagBlock(const a_int starti, const a_real *const buffer, const a_int dummy);

	/// Scales all values by a constact scalar
	void scaleAll(const a_real factor);

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

	a_int dim() const { return m->gnelem()*bs; }
	
	/// Prints diagonal, L or U blocks depending on the argument
	void printDiagnostic(const char choice) const;
};

}

namespace acfd {

/// Vector or matrix addition
/** z <- pz + qx.
 * \param[in] N The length of the vectors
 */
inline void axpby(const a_int N, const a_real p, a_real *const __restrict z, 
	const a_real q, const a_real *const x)
{
	//a_real *const zz = &z(0,0); const a_real *const xx = &x(0,0);
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		z[i] = p*z[i] + q*x[i];
	}
}

/** z <- pz + qx + ry for vectors and matrices
 */
inline void axpbypcz(const a_int N, const a_real p, a_real *const z, 
	const a_real q, const a_real *const x,
	const a_real r, const a_real *const y)
{
	//a_real *const zz = &z(0,0); const a_real *const xx =&x(0,0); const a_real *const yy = &y(0,0);
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		z[i] = p*z[i] + q*x[i] + r*y[i];
	}
}

/// Dot product of vectors or `double dot' product of matrices
inline a_real dot(const a_int N, const a_real *const a, 
	const a_real *const b)
{
	a_real sum = 0;
#pragma omp parallel for simd default(shared) reduction(+:sum)
	for(a_int i = 0; i < N; i++)
		sum += a[i]*b[i];

	return sum;
}


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
	/** \param[in] r The right hand side vector
	 * \param [in|out] z Contains the solution
	 */
	virtual void apply(const a_real *const r, 
			a_real *const z) = 0;
};

/// Do-nothing preconditioner
/** The preconditioner is the identity matrix.
 */
template <short nvars>
class NoPrec : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	NoPrec(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op)
	{ }
	
	void compute()
	{ }
	
	void apply(const a_real *const r, 
			a_real *const z)
	{
#pragma omp parallel for simd default(shared)
		for(a_int i = 0; i < A->dim(); i++)
			z[i] = r[i];
	}
};

/// Jacobi preconditioner
template <short nvars>
class Jacobi : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	Jacobi(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }
	
	void compute() {
		A->precJacobiSetup();
	}

	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precJacobiApply(r, z);
	}
};

/// Symmetric Gauss-Seidel preconditioner
template <short nvars>
class SGS : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	SGS(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) 
	{
		A->allocTempVector();
	}

	/// Sets D,L,U and inverts each D
	void compute() {
		A->precJacobiSetup();
	}

	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precSGSApply(r, z);
	}
};

/// ILU0 preconditioner
template <short nvars>
class ILU0 : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	ILU0(LinearOperator<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }

	/// Sets D,L,U and computes the ILU factorization
	void compute() {
		A->precILUSetup();
	}
	
	/// Solves Mz=r, where M is the preconditioner
	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precILUApply(r, z);
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

/// Abstract preconditioned iterative solver
class IterativeSolverBase : public LinearSolver
{
protected:
	int maxiter;                                  ///< Max number of iterations
	double tol;                                   ///< Tolerance
	mutable double walltime;                      ///< Stores wall-clock time measurement of solver
	mutable double cputime;                       ///< Stores CPU time measurement of the solver

public:
	IterativeSolverBase(const UMesh2dh* const mesh);

	//virtual ~IterativeSolverBase();
	
	/// Set tolerance and max iterations
	void setParams(const double toler, const int maxits) {
		maxiter = maxits; tol = toler;
	}
	
	/// Sets time accumulators to zero
	void resetRunTimes() {
		walltime = 0; cputime = 0;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}
};

/// Preconditioned iterative solver that relies on a stored LHS matrix
/** The template parameter nvars is the block size we want to use.
 * In a finite volume setting, the natural choice is the number of physical variables
 * or the number of PDEs in the system.
 */
template <short nvars>
class IterativeSolver : public IterativeSolverBase
{
protected:
	LinearOperator<a_real,a_int> *const A;        ///< The LHS matrix context

	/// Preconditioner context
	Preconditioner<nvars> *const prec;

public:
	IterativeSolver(const UMesh2dh* const mesh, 
			LinearOperator<a_real,a_int>* const mat, 
			Preconditioner<nvars> *const precond);

	//virtual ~IterativeSolver();

	/// Compute the preconditioner
	virtual void setupPreconditioner();

	/// Solves the linear system A du = -r
	/** Note that usually, the two arguments cannot alias each other.
	 * \param[in] res The residual vector stored as a 2D array of size nelem x nvars 
	 * (nelem x 4 for 2D Euler)
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const MVector& res, 
			MVector& __restrict du) const = 0;
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
	RichardsonSolver(const UMesh2dh* const mesh, 
			LinearOperator<a_real,a_int>* const mat, 
			Preconditioner<nvars> *const precond);

	/** \param[in] res The right hand side vector
	 * \param[in] du The solution vector which is assumed to contain an initial solution
	 *
	 * \warning The two arguments must not alias each other.
	 */
	int solve(const MVector& res, 
		MVector& __restrict du) const;
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
	BiCGSTAB(const UMesh2dh* const mesh, 
			LinearOperator<a_real,a_int>* const mat, 
			Preconditioner<nvars> *const precond);

	int solve(const MVector& res, 
		MVector& __restrict du) const;
};

/// Restarted GMRES
template <short nvars>
class GMRES : public IterativeSolver<nvars>
{
	using IterativeSolver<nvars>::m;
	using IterativeSolver<nvars>::A;
	using IterativeSolver<nvars>::maxiter;
	using IterativeSolver<nvars>::tol;
	using IterativeSolver<nvars>::walltime;
	using IterativeSolver<nvars>::cputime;
	using IterativeSolver<nvars>::prec;
	
	/// Number of Krylov subspace vectors to store
	int mrestart;

public:
	GMRES(const UMesh2dh* const mesh, 
			LinearOperator<a_real,a_int>* const mat, 
			Preconditioner<nvars> *const precond,
			int m_restart);

	int solve(const MVector& res, 
		MVector& __restrict du) const;
};

/// Base class for matrix-free solvers
/** Note that subclasses are matrix-free only with regard to the top-level solver,
 * usually a Krylov subspace solver. The preconditioning matrix is still computed and stored.
 */
template <short nvars>
class MFIterativeSolver : public IterativeSolverBase
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

public:
	MFIterativeSolver(const UMesh2dh* const mesh, Preconditioner<nvars> *const precond,
		Spatial<nvars> *const spatial);

	//virtual ~MFIterativeSolver();

	/// Sets D,L,U for preconditioner
	virtual void setupPreconditioner();

	/// Solves the linear system A du = -r
	/** \param[in] u The state at which the Jacobian and RHS res have been computed
	 * \param[in] res The residual vector stored as a 2D array of size nelem x nvars 
	 * (nelem x 4 for 2D Euler)
	 * \param aux Temporary storage needed for matrix-free evaluation of Jacobian-vector products
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const MVector& __restrict__ u, 
		const amat::Array2d<a_real>& dtm,
		const MVector& __restrict__ res, 
		MVector& __restrict__ aux,
		MVector& __restrict__ du) const = 0;
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

	int solve(const MVector& __restrict__ u, 
		const amat::Array2d<a_real>& dtm,
		const MVector& __restrict__ res, 
		MVector& __restrict__ aux,
		MVector& __restrict__ du) const;
};

/// Sets up storage preallocation for sparse matrix formats
/** \param[in] m Mesh context
 * \param[in] mat_type A character which selects the matrix storage scheme for the Jacobian.
 *            Possible values: 'p' (point CSR storage), 'b' (block CSR storage) 
 *            or 'd' ('DLU' storage)
 * \param[in|out] A The matrix to pre-allocate for
 */
template <short nvars>
void setupMatrixStorage(const UMesh2dh *const m, const char mattype,
		LinearOperator<a_real,a_int> *const A);

}
#endif

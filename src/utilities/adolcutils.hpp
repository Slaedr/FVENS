/** \file
 * \brief Utilities to inter-operate with ADOL-C adoubles
 */

#ifndef FVENS_ADOLC_UTILS_H
#define FVENS_ADOLC_UTILS_H

#include <petscvec.h>

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

#include "aerrorhandling.hpp"

namespace fvens {

/// Returns the value of the input as a_real
template <typename scalar>
static inline a_real getvalue(const scalar x);

/// Returns a PETSc Vec as a raw array
template <typename scalar>
static inline scalar *getVecAsArray(Vec x);

/// Returns a PETSc Vec as a const raw array
template <typename scalar>
static inline const scalar *getVecAsReadOnlyArray(Vec x);

/// Restore an array back to a PETSc Vec
template <typename scalar>
static inline void restoreArraytoVec(Vec x, scalar **arr);

/// Restore a const array back to a PETSc Vec
template <typename scalar>
static inline void restoreReadOnlyArraytoVec(Vec x, const scalar **arr);

// Trivial implementations

template <>
a_real getvalue(const a_real x) {
	return x;
}

template <>
a_real *getVecAsArray(Vec x)
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	a_real *xarr;
	int ierr = VecGetArray(x, &xarr);
	petsc_throw(ierr, "Could not get array from Vec!");
	return xarr;
}

template <>
const a_real *getVecAsReadOnlyArray(Vec x)
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	const a_real *xarr;
	int ierr = VecGetArrayRead(x, &xarr);
	petsc_throw(ierr, "Could not get array from Vec!");
	return xarr;
}

template <>
void restoreArraytoVec(Vec x, a_real **arr)
{
	int ierr = VecRestoreArray(x, arr);
	petsc_throw(ierr, "Could not restore array to Vec!");
}

template <>
void restoreReadOnlyArraytoVec(Vec x, const a_real **arr)
{
	int ierr = VecRestoreArrayRead(x, arr);
	petsc_throw(ierr, "Could not restore const array to Vec!");
}

// ADOL-C implementations (incomplete!)

#ifdef USE_ADOLC

template <>
a_real getvalue(const adouble x) {
	return x.value();
}

/// The adouble version creates a deep copy of the local portion of the Vec
/** Note that we do not call VecRestoreArray. This is on purpose - PETSc should think it's in use
 * until \ref restoreArraytoVec is called.
 */
template <>
adouble *getVecAsArray(Vec x)
{
	static_assert(std::is_same<PetscScalar,double>::value,
	              "PETSc scalar must be double for use with ADOL-C.");
	double *xarr;
	int ierr = VecGetArray(x, &xarr);
	petsc_throw(ierr, "Could not get array from Vec!");

	PetscInt sz = 0;
	ierr = VecGetLocalSize(x, &sz);
	petsc_throw(ierr, "Could not get local Vec size!");

	adouble *adarray = new adouble[sz];

	// copy and activate
	for(PetscInt i = 0; i < sz; i++)
		adarray[i] <<= xarr[i];

	return adarray;
}

template <>
void restoreArraytoVec(Vec x, adouble **arr)
{
	PetscInt sz = 0;
	int ierr = VecGetLocalSize(x, &sz);
	petsc_throw(ierr, "Could not get Vec local size!");

	PetscScalar *xarr = new PetscScalar[sz];
	for(PetscInt i = 0; i < sz; i++)
		xarr[i] = (*arr)[i];

	ierr = VecRestoreArray(x, &xarr);
	petsc_throw(ierr, "Could not restore array to Vec!");

	delete [] xarr;
	delete [] *arr;
	*arr = NULL;
}

#endif


}
#endif

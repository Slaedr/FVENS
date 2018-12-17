/** \file
 * \brief Implementation of some functions to make PETSc more convenient in some way, eg., usage
 *   with ADOL-C.
 */

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

#include "petscutils.hpp"
#include "aconstants.hpp"
#include "utilities/aerrorhandling.hpp"

namespace fvens {

template <>
VecHandler<a_real>::VecHandler(Vec x) : vec{x}, data{NULL}, cdata{NULL}
{ }

template <>
a_real *VecHandler<a_real>::getVecAsArray()
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	int ierr = VecGetArray(vec, &data);
	petsc_throw(ierr, "Could not get array from Vec!");
	return data;
}

template <>
void VecHandler<a_real>::restoreArrayToVec(a_real **const arr)
{
	int ierr = VecRestoreArray(vec, arr);
	petsc_throw(ierr, "Could not restore array to Vec!");
	*arr = nullptr;
}

template <>
const a_real *VecHandler<a_real>::getVecAsReadOnlyArray() const
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	int ierr = VecGetArrayRead(vec, &cdata);
	petsc_throw(ierr, "Could not get array from Vec!");
	return cdata;
}

template <>
void VecHandler<a_real>::restoreReadOnlyArrayToVec(const a_real **const arr) const
{
	int ierr = VecRestoreArrayRead(vec, arr);
	petsc_throw(ierr, "Could not restore const array to Vec!");
	*arr = nullptr;
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

#ifdef USE_ADOLC

/// The adouble version creates a deep copy of the local portion of the Vec
/** Note that we do not call VecRestoreArray. This is on purpose - PETSc should think it's in use
 * until \ref restoreArraytoVec is called.
 */
template <>
adouble *getVecAsArray(Vec x)
{
	static_assert(std::is_same<PetscScalar,double>::value,
	              "PETSc scalar must be double for use with ADOL-C.");
	PetscScalar *xarr;
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

/** Purposefully does not delete the array which is finally given to VecRestoreArray.
 */
template <>
void restoreArraytoVec(Vec x, adouble **arr)
{
	PetscInt sz = 0;
	int ierr = VecGetLocalSize(x, &sz);
	petsc_throw(ierr, "Could not get Vec local size!");

	PetscScalar *xarr;
	ierr = PetscMalloc1(sz, &xarr);
	for(PetscInt i = 0; i < sz; i++)
		xarr[i] = (*arr)[i];

	ierr = VecRestoreArray(x, &xarr);
	petsc_throw(ierr, "Could not restore array to Vec!");

	delete [] *arr;
	*arr = NULL;
}

#endif

}

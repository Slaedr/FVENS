/** \file
 * \brief Implementation of some functions to make PETSc more convenient in some way, eg., usage
 *   with ADOL-C.
 */

#include <iostream>
#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

#include "petscutils.hpp"
#include "aconstants.hpp"
#include "utilities/aerrorhandling.hpp"

namespace fvens {

template <>
MutableVecHandler<a_real>::MutableVecHandler(Vec x) : vec{x}
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	int ierr = VecGetArray(vec, &data);
	petsc_throw(ierr, "Could not get local raw array!");
	sdata = data;
}

template <>
MutableVecHandler<a_real>::~MutableVecHandler()
{
	sdata = nullptr;
	int ierr = VecRestoreArray(vec, &data);
	if(ierr)
		std::cout << " Could not restore PETSc Vec!" << std::endl;
}

template <>
a_real *MutableVecHandler<a_real>::getArray()
{
	return sdata;
}

template <>
ConstVecHandler<a_real>::ConstVecHandler(Vec x) : vec{x}
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	int ierr = VecGetArrayRead(vec, &data);
	petsc_throw(ierr, "Could not get local raw array!");
	sdata = data;
}

template <>
ConstVecHandler<a_real>::~ConstVecHandler()
{
	sdata = nullptr;
	int ierr = VecRestoreArrayRead(vec, &data);
	if(ierr)
		std::cout << " Could not restore PETSc Vec!" << std::endl;
}

template <>
const a_real *ConstVecHandler<a_real>::getArray() const
{
	return sdata;
}

template <>
MutableGhostedVecHandler<a_real>::MutableGhostedVecHandler(Vec x)
	: MutableVecHandler(x)
{
	int ierr = VecGhostGetLocalForm(vec, &localvec);
	petsc_throw(ierr, "Could not get local form!");
	ierr = VecGetArray(localvec, &data);
	petsc_throw(ierr, "Could not get local raw array!");
	sdata = data;
}

template <>
MutableGhostedVecHandler<a_real>::~MutableGhostedVecHandler()
{
	sdata = nullptr;
	int ierr = VecRestoreArray(localvec, &data);
	if(ierr)
		std::cout << " Could not restore array to Vec!" << std::endl;
	ierr = VecGhostRestoreLocalForm(vec, &localvec);
	if(ierr)
		std::cout << " Could not restore local PETSc Vec!" << std::endl;
}

template <>
ConstGhostedVecHandler<a_real>::ConstGhostedVecHandler(Vec x)
	: ConstVecHandler(x)
{
	int ierr = VecGhostGetLocalForm(vec, &localvec);
	petsc_throw(ierr, "Could not get local form!");
	ierr = VecGetArrayRead(localvec, &data);
	petsc_throw(ierr, "Could not get local raw array!");
	sdata = data;
}

template <>
ConstGhostedVecHandler<a_real>::~ConstGhostedVecHandler()
{
	sdata = nullptr;
	int ierr = VecRestoreArrayRead(localvec, &data);
	if(ierr)
		std::cout << " Could not restore array to Vec!" << std::endl;
	ierr = VecGhostRestoreLocalForm(vec, &localvec);
	if(ierr)
		std::cout << " Could not restore local PETSc Vec!" << std::endl;
}

#if 0
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
#endif

}

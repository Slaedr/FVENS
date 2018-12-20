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

template <typename scalar>
MutableVecHandler<scalar>::MutableVecHandler() : vec{NULL}, data{NULL}, sdata{nullptr}
{ }

template <typename scalar>
MutableVecHandler<scalar>::MutableVecHandler(Vec x) : vec{NULL}
{
	static_assert(std::is_convertible<PetscScalar,scalar>::value,
	              "a_real type does not match PETSc scalar!");
	setVec(x);
}

template <typename scalar>
MutableVecHandler<scalar>::~MutableVecHandler()
{
	restore();
}

template <typename scalar>
void MutableVecHandler<scalar>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGetArray(vec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <typename scalar>
void MutableVecHandler<scalar>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArray(vec, &data);
		if(ierr)
			std::cout << " Could not restore PETSc Vec!" << std::endl;
		vec = NULL;
	}
}

template <typename scalar>
scalar *MutableVecHandler<scalar>::getArray()
{
	return sdata;
}

template class MutableVecHandler<a_real>;

template <typename scalar>
ConstVecHandler<scalar>::ConstVecHandler() : vec{NULL}, data{NULL}, sdata{nullptr}
{ }

template <typename scalar>
ConstVecHandler<scalar>::ConstVecHandler(Vec x) : vec{NULL}
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	setVec(x);
}

template <typename scalar>
ConstVecHandler<scalar>::~ConstVecHandler()
{
	restore();
}

template <typename scalar>
void ConstVecHandler<scalar>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGetArrayRead(vec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <typename scalar>
void ConstVecHandler<scalar>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArrayRead(vec, &data);
		if(ierr)
			std::cout << " Could not restore PETSc Vec!" << std::endl;
		vec = NULL;
	}
}

template <typename scalar>
const scalar *ConstVecHandler<scalar>::getArray() const
{
	return sdata;
}

template class ConstVecHandler<a_real>;

template <typename scalar>
MutableGhostedVecHandler<scalar>::MutableGhostedVecHandler()
	: MutableVecHandler<scalar>(), localvec{NULL}
{ }

template <typename scalar>
MutableGhostedVecHandler<scalar>::MutableGhostedVecHandler(Vec x)
	: MutableVecHandler<scalar>(), localvec{NULL}
{
	setVec(x);
}

template <typename scalar>
MutableGhostedVecHandler<scalar>::~MutableGhostedVecHandler()
{
	restore();
}

template <typename scalar>
void MutableGhostedVecHandler<scalar>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGhostGetLocalForm(vec, &localvec);
		petsc_throw(ierr, "Could not get local form!");
		ierr = VecGetArray(localvec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <typename scalar>
void MutableGhostedVecHandler<scalar>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArray(localvec, &data);
		if(ierr)
			std::cout << " Could not restore array to Vec!" << std::endl;
		ierr = VecGhostRestoreLocalForm(vec, &localvec);
		if(ierr)
			std::cout << " Could not restore local PETSc Vec!" << std::endl;
		vec = NULL;
		localvec = NULL;
	}
}

template class MutableGhostedVecHandler<a_real>;

template <typename scalar>
ConstGhostedVecHandler<scalar>::ConstGhostedVecHandler() : ConstVecHandler<scalar>(), localvec{NULL}
{ }

template <typename scalar>
ConstGhostedVecHandler<scalar>::ConstGhostedVecHandler(Vec x) : ConstVecHandler<scalar>(),localvec{NULL}

{
	setVec(x);
}

template <typename scalar>
ConstGhostedVecHandler<scalar>::~ConstGhostedVecHandler()
{
	restore();
}

template <typename scalar>
void ConstGhostedVecHandler<scalar>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGhostGetLocalForm(vec, &localvec);
		petsc_throw(ierr, "Could not get local form!");
		ierr = VecGetArrayRead(localvec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <typename scalar>
void ConstGhostedVecHandler<scalar>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArrayRead(localvec, &data);
		if(ierr)
			std::cout << " Could not restore array to Vec!" << std::endl;
		ierr = VecGhostRestoreLocalForm(vec, &localvec);
		if(ierr)
			std::cout << " Could not restore local PETSc Vec!" << std::endl;
		vec = NULL;
		localvec = NULL;
	}
}

template class ConstGhostedVecHandler<a_real>;

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

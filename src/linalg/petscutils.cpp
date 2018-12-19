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
MutableVecHandler<a_real>::MutableVecHandler() : vec{NULL}, data{NULL}, sdata{nullptr}
{ }

template <>
MutableVecHandler<a_real>::MutableVecHandler(Vec x) : vec{NULL}
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	setVec(x);
}

template <>
MutableVecHandler<a_real>::~MutableVecHandler()
{
	restore();
}

template <>
void MutableVecHandler<a_real>::setVec(Vec x)
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

template <>
void MutableVecHandler<a_real>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArray(vec, &data);
		if(ierr)
			std::cout << " Could not restore PETSc Vec!" << std::endl;
		vec = NULL;
		localvec = NULL;
	}
}

template <>
a_real *MutableVecHandler<a_real>::getArray()
{
	return sdata;
}

template <>
ConstVecHandler<a_real>::ConstVecHandler() : vec{NULL}, data{NULL}, sdata{nullptr}
{ }

template <>
ConstVecHandler<a_real>::ConstVecHandler(Vec x) : vec{NULL}
{
	static_assert(std::is_same<PetscScalar,a_real>::value, "a_real type does not match PETSc scalar!");
	setVec(x);
}

template <>
ConstVecHandler<a_real>::~ConstVecHandler()
{
	restore();
}

template <>
void ConstVecHandler<a_real>::setVec(Vec x)
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

template <>
void ConstVecHandler<a_real>::restore()
{
	sdata = nullptr;
	if(vec) {
		int ierr = VecRestoreArrayRead(vec, &data);
		if(ierr)
			std::cout << " Could not restore PETSc Vec!" << std::endl;
		vec = NULL;
		localvec = NULL;
	}
}

template <>
const a_real *ConstVecHandler<a_real>::getArray() const
{
	return sdata;
}

template <>
MutableGhostedVecHandler<a_real>::MutableGhostedVecHandler() : vec{NULL}, data{NULL},
                                                               sdata{nullptr}, localvec{NULL}
{ }

template <>
MutableGhostedVecHandler<a_real>::MutableGhostedVecHandler(Vec x) : vec{NULL}, data{NULL},
                                                                    sdata{nullptr}, localvec{NULL}
{
	setVec(x);
}

template <>
MutableGhostedVecHandler<a_real>::~MutableGhostedVecHandler()
{
	restore();
}

template <>
void MutableGhostedVecHandler<a_real>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGhostGetLocalForm(vec, &localvec);
		petsc_throw(ierr, "Could not get local form!");
		int ierr = VecGetArray(localvec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <>
void MutableGhostedVecHandler<a_real>::restore()
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

template <>
ConstGhostedVecHandler<a_real>::ConstGhostedVecHandler() : vec{NULL}, data{NULL},
                                                           sdata{nullptr}, localvec{NULL}
{ }

template <>
ConstGhostedVecHandler<a_real>::ConstGhostedVecHandler(Vec x) : vec{NULL}, data{NULL},
                                                                sdata{nullptr}, localvec{NULL}

{
	setVec(x);
}

template <>
ConstGhostedVecHandler<a_real>::~ConstGhostedVecHandler()
{
	restore();
}

template <>
void ConstGhostedVecHandler<a_real>::setVec(Vec x)
{
	if(!vec) {
		vec = x;
		int ierr = VecGhostGetLocalForm(vec, &localvec);
		petsc_throw(ierr, "Could not get local form!");
		int ierr = VecGetArrayRead(localvec, &data);
		petsc_throw(ierr, "Could not get local raw array!");
		sdata = data;
	}
	else
		throw std::runtime_error("MutableVecHandler already has a Vec attached!");
}

template <>
void ConstGhostedVecHandler<a_real>::restore()
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

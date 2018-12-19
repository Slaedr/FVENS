/** \file
 * \brief Some convenience functions that wrap PETSc functionality for various reasons.
 */

#ifndef FVENS_PETSCUTILS_H
#define FVENS_PETSCUTILS_H

#include <petscvec.h>

namespace fvens {

/// Maintains a native array corresponding to a PETSc Vec and provides access
template <typename scalar>
class MutableVecHandler
{
public:
	MutableVecHandler(Vec x);
	virtual ~MutableVecHandler();
	virtual scalar *getArray();

protected:
	const Vec vec;
	PetscScalar *data;
	scalar *sdata;
};

/// Maintains an immutable native array corresponding to a PETSc Vec and provides access
template <typename scalar>
class ConstVecHandler
{
public:
	ConstVecHandler(Vec x);
	virtual ~ConstVecHandler();
	virtual const scalar *getArray() const;

protected:
	const Vec vec;
	const PetscScalar *data;
	const scalar *sdata;
};

/// Maintains an immutable native array corresponding to a ghosted PETSc Vec and provides access
template <typename scalar>
class MutableGhostedVecHandler : public MutableVecHandler<scalar>
{
public:
	MutableGhostedVecHandler(Vec x);
	~MutableGhostedVecHandler();
	using MutableVecHandler<scalar>::getArray;

protected:
	using MutableVecHandler<scalar>::vec;
	using MutableVecHandler<scalar>::data;
	using MutableVecHandler<scalar>::sdata;
	Vec localvec;
};

/// Maintains an immutable native array corresponding to a ghosted PETSc Vec and provides access
template <typename scalar>
class ConstGhostedVecHandler : public ConstVecHandler<scalar>
{
public:
	ConstGhostedVecHandler(Vec x);
	~ConstGhostedVecHandler();
	using ConstVecHandler<scalar>::getArray;

protected:
	using ConstVecHandler<scalar>::vec;
	using ConstVecHandler<scalar>::data;
	using ConstVecHandler<scalar>::sdata;
	Vec localvec;
};

}

#endif

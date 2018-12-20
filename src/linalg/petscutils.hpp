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
	MutableVecHandler();
	~MutableVecHandler();
	/// Set a Vec to handle - only works if the object is not already associated with a Vec
	void setVec(Vec x);
	void restore();
	scalar *getArray();

protected:
	Vec vec;
	PetscScalar *data;
	scalar *sdata;
};

/// Maintains an immutable native array corresponding to a PETSc Vec and provides access
template <typename scalar>
class ConstVecHandler
{
public:
	ConstVecHandler(Vec x);
	ConstVecHandler();
	~ConstVecHandler();
	/// Set a Vec to handle - only works if the object is not already associated with a Vec
	void setVec(Vec x);
	void restore();
	const scalar *getArray() const;

protected:
	Vec vec;
	const PetscScalar *data;
	const scalar *sdata;
};

/// Maintains an immutable native array corresponding to a ghosted PETSc Vec and provides access
template <typename scalar>
class MutableGhostedVecHandler : public MutableVecHandler<scalar>
{
public:
	MutableGhostedVecHandler();
	MutableGhostedVecHandler(Vec x);
	~MutableGhostedVecHandler();
	void setVec(Vec x);
	void restore();
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
	ConstGhostedVecHandler();
	ConstGhostedVecHandler(Vec x);
	~ConstGhostedVecHandler();
	void setVec(Vec x);
	void restore();
	using ConstVecHandler<scalar>::getArray;

protected:
	using ConstVecHandler<scalar>::vec;
	using ConstVecHandler<scalar>::data;
	using ConstVecHandler<scalar>::sdata;
	Vec localvec;
};

}

#endif

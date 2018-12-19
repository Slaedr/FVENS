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
	~MutableVecHandler();
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
	~ConstVecHandler();
	virtual const scalar *getArray() const;

protected:
	const Vec vec;
	const PetscScalar *data;
	const scalar *sdata;
};

// template <typename scalar>
// class GhostedVecHandler : public VecHandler<scalar>
// {
// public:
// 	GhostedVecHandler(Vec x);
// 	scalar *getVecAsArray();
// 	const scalar *getVecAsReadOnlyArray() const ;
// 	void restoreArrayToVec(scalar **const arr) ;
// 	void restoreReadOnlyArrayToVec(const scalar **const arr) const;

// protected:
// 	using VecHandler<scalar>::vec;
// 	using VecHandler<scalar>::data;
// 	using VecHandler<scalar>::cdata;
// 	mutable Vec local;
// };

}

#endif

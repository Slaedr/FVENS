/** \file
 * \brief Some convenience functions that wrap PETSc functionality for various reasons.
 */

#ifndef FVENS_PETSCUTILS_H
#define FVENS_PETSCUTILS_H

#include <petscvec.h>

namespace fvens {

/// Returns a PETSc Vec as a raw array
template <typename scalar>
scalar *getVecAsArray(Vec x);

/// Returns a PETSc Vec as a const raw array
template <typename scalar>
const scalar *getVecAsReadOnlyArray(Vec x);

/// Restore an array back to a PETSc Vec
template <typename scalar>
void restoreArraytoVec(Vec x, scalar **arr);

/// Restore a const array back to a PETSc Vec
template <typename scalar>
void restoreReadOnlyArraytoVec(Vec x, const scalar **arr);

/// Get the local portion of a PETSc ghosted vec as an array
template <typename scalar>
scalar *getGhostedVecLocalArray(Vec x);

template <typename scalar>
class VecHandler
{
public:
	VecHandler(Vec x);
	virtual scalar *getVecAsArray();
	virtual const scalar *getVecAsReadOnlyArray() const ;
	virtual void restoreArrayToVec(scalar **const arr) ;
	virtual void restoreReadOnlyArrayToVec(const scalar **const arr) const;

protected:
	Vec vec;
	mutable PetscScalar *data;
	mutable const PetscScalar *cdata;
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

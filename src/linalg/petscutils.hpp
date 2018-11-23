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

}

#endif

/** \file
 * \brief Extra utilities which are currently not used
 */

#ifndef FVENS_EXTRA_MESHUTILS_H
#define FVENS_EXTRA_MESHUTILS_H

#include "amesh2dh.hpp"

namespace fvens {

/** \brief Reads 'domn' format
 *
 * \note Make sure nfael and nnofa are mentioned after ndim and nnode in the mesh file.
 * \deprecated Please use Gmsh format instead.
 */
void readDomn(const std::string mfile, UMesh2dh<scalar>& mesh);

/** \brief Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
 *
 * \note Make sure to execute [compute_topological()](@ref compute_topological)
 * before calling this function.
 */
template <typename scalar>
UMesh2dh<scalar> convertLinearToQuadratic(const UMesh2dh<scalar>& m);

/// Converts quads in a mesh to triangles
template <typename scalar>
UMesh2dh<scalar> convertQuadToTri(const UMesh2dh<scalar>& m);

}

#endif

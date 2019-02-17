/** \file
 * \brief Some useful orderings for cells in the mesh
 * \author Aditya Kashi
 * \date 2019-02
 */

#ifndef FVENS_MESH_ORDERING_H
#define FVENS_MESH_ORDERING_H

#include "amesh2dh.hpp"

namespace fvens {

template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m);

}

#endif

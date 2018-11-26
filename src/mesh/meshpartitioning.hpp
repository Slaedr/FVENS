/** \file
 * \brief Routines for distributing a mesh among available processes
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MESH_PARTITIONING_H
#define FVENS_MESH_PARTITIONING_H

#include "amesh2dh.hpp"

namespace fvens {

/// Given the global mesh on rank 0, partitions it in a trivial manner
UMesh2dh<a_real> partitionMeshTrivial(const MeshData& global_mesh);

/// Populates this process's share of mesh arrays from the global arrays
/** Assumptions: gm's integers are available on all ranks.
 * gm's arrays and the array glbElemDist are only available on rank 0.
 */
void splitMeshArrays(const MeshData& gm, const std::vector<int>& glbElemDist,
                     UMesh2dh<a_real>& lm);

//void partitionMesh(UMesh2dh<a_real>& mesh);

}

#endif

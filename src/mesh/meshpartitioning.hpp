/** \file
 * \brief Routines for distributing a mesh among available processes
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MESH_PARTITIONING_H
#define FVENS_MESH_PARTITIONING_H

#include "amesh2dh.hpp"

namespace fvens {

/// Memory-inefficient partitioner that assumes the global mesh is available on all partitions
class ReplicatedGlobalMeshPartitioner
{
public:
	/// Given the global mesh, returns the localized mesh on this rank
	virtual UMesh2dh<a_real> partition(const UMesh2dh<a_real>& global_mesh) const = 0;

protected:
	/// Given a partition of the elements and the global mesh, returns the localized mesh on this rank
	UMesh2dh<a_real> restrictMeshToPartitions(const UMesh2dh<a_real>& global_mesh,
	                                          const std::vector<int>& elem_partition) const;
};

/// Partitions the mesh trivially based on the initial global element ordering
class TrivialReplicatedGlobalMeshPartitioner : public ReplicatedGlobalMeshPartitioner
{
public:
	UMesh2dh<a_real> partition(const UMesh2dh<a_real>& global_mesh) const;
};

/// Given the global mesh on rank 0, partitions it in a trivial manner
UMesh2dh<a_real> partitionMeshTrivial(const MeshData& global_mesh);

/// Populates this process's share of mesh arrays from the global arrays
/** Assumptions: gm's integers are available on all ranks.
 * gm's arrays and the array glbElemDist are only available on rank 0.
 */
void splitMeshArrays(const MeshData& gm, const std::vector<int>& glbElemDist,
                     UMesh2dh<a_real>& lm);

}

#endif

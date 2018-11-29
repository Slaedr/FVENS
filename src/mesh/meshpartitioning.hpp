/** \file
 * \brief Routines for distributing a mesh among available processes
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MESH_PARTITIONING_H
#define FVENS_MESH_PARTITIONING_H

#include <vector>
#include <map>
#include "amesh2dh.hpp"

namespace fvens {

/// Memory-inefficient partitioner that assumes the global mesh is available on all partitions
class ReplicatedGlobalMeshPartitioner
{
public:
	/// Sets the global mesh to partition
	/** The mesh must be setup with all topological connectivity structures before passing here.
	 */
	ReplicatedGlobalMeshPartitioner(const UMesh2dh<a_real>& global_mesh);
	
	/// Given the global mesh, computes the distribution of the elements
	virtual void compute_partition() = 0;

	/// Given a partition of the elements and the global mesh, returns the localized mesh on this rank
	UMesh2dh<a_real> restrictMeshToPartitions() const;

protected:
	/// Extracts the local element-node connectivity info from the global mesh into the argument mesh
	/** Important: The local inpoel still contains global node indices at the end of this.
	 * Also populates the local nnode and nfael arrays.
	 * \param[in,out] The local mesh to update inpoel, nnode, nfael, and vol_regions in.
	 *   These arrays must be pre-allocated. Local number of elements nelem must be set beforehand.
	 * \return the local-to-global element map
	 */
	std::vector<a_int> extractInpoel(UMesh2dh<a_real>& local_mesh) const;

	/// Extracts point coordinates required by this rank
	/** \param lm The local mesh with inpoel pre-computed in terms of the global point indices
	 *    \ref extractInpoel
	 * \param[out] locpoints The global point indices of local points
	 * \return The local point indices of global points required by this rank
	 */
	std::map<a_int,a_int> extractPointCoords(UMesh2dh<a_real>& lm,
	                                         std::vector<a_int>& locpoints) const;

	/// Extracts the physical boundary faces which are a part of the subdomain on this rank
	/** \param pointGlobal2Local Map from global to local indices for points present in this rank.
	 */
	void extractbfaces(const std::map<a_int,a_int>& pointGlobal2Local, UMesh2dh<a_real>& lm) const;

	/// Marks cells of this subdomain which are adjacent to physical boundary faces
	/** As a by-product, also computes the number of physical boundary points in the local mesh in
	 * nbpoin.
	 */
	std::vector<bool> markLocalBoundaryCells(const UMesh2dh<a_real>& lm) const;

	/// The global mesh to partition
	const UMesh2dh<a_real>& gm;

	/// The distribution of elements among the ranks
	/** Supposed to hold the rank where each element of the global mesh is supposed to go
	 */
	std::vector<int> elemdist;
};

/// Partitions the mesh trivially based on the initial global element ordering
class TrivialReplicatedGlobalMeshPartitioner : public ReplicatedGlobalMeshPartitioner
{
public:
	/// Sets the global mesh to partition
	/** The mesh must be setup with all topological connectivity structures before passing here.
	 */
	TrivialReplicatedGlobalMeshPartitioner(const UMesh2dh<a_real>& global_mesh);

	void compute_partition();
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

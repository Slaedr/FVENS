/** \file
 * \brief Routines for distributing a mesh among available processes
 * \author Aditya Kashi
 * \date 2018-11
 */

#ifndef FVENS_MESH_PARTITIONING_H
#define FVENS_MESH_PARTITIONING_H

#include <vector>
#include <map>
#include <tuple>
#include "mesh.hpp"

namespace fvens {

/// Memory-inefficient partitioner that assumes the global mesh is available on all partitions
class ReplicatedGlobalMeshPartitioner
{
public:
	/// Sets the global mesh to partition
	/** The mesh must be setup with the element adjacency list (esuel) before passing here.
	 */
	ReplicatedGlobalMeshPartitioner(const UMesh<freal,NDIM>& global_mesh);
	
	/// Given the global mesh, computes the distribution of the elements
	virtual void compute_partition() = 0;

	/// Computes the localized mesh on this rank given a partition of the cells of the global mesh
	UMesh<freal,NDIM> restrictMeshToPartitions() const;

	/// Check whether the global face numbering stored in the local mesh argument gives consistent
	///  left elements in the actual global mesh
	/** \param[in] lmesh A restricted mesh with face structure computed
	 */
	bool checkConnFaces(const UMesh<freal,NDIM>& lmesh) const;

protected:
	/// Extracts the local element-node connectivity info from the global mesh into the argument mesh
	/** Important: The local inpoel still contains global node indices at the end of this.
	 * Also populates the local nnode and nfael arrays.
	 * \param[in,out] The local mesh to update inpoel, nnode, nfael, and vol_regions in.
	 *   These arrays must be pre-allocated. Local number of elements nelem must be set beforehand.
	 * \return the local-to-global element map
	 */
	std::vector<fint> extractInpoel(UMesh<freal,NDIM>& local_mesh) const;

	/// Extracts point coordinates required by this rank
	/** \param lm The local mesh with inpoel pre-computed in terms of the global point indices
	 *    \ref extractInpoel
	 * \param[out] locpoints The global point indices of local points
	 * \return The local point indices of global points required by this rank
	 */
	std::map<fint,fint> extractPointCoords(UMesh<freal,NDIM>& lm,
	                                       std::vector<fint>& locpoints) const;

	/// Extracts the physical boundary faces which are a part of the subdomain on this rank
	/** \param pointGlobal2Local Map from global to local indices for points present in this rank.
	 */
	void extractbfaces(const std::map<fint,fint>& pointGlobal2Local, UMesh<freal,NDIM>& lm) const;

	/// Marks points of this subdomain which are adjacent to physical boundary faces
	/** As a by-product, also computes the number of physical boundary points in the local mesh in
	 * nbpoin.
	 * \return The vector of point marks
	 */
	std::vector<bool> markLocalPhysicalBoundaryPoints(const UMesh<freal,NDIM>& lm) const;

	/// Returns a list of containing, for each local element, face E-indices that are connectivity
	///  faces, if any
	/** \param[in] isPhyBounPoint For each local point, whether or not it lies on a physical boundary
	 */
	std::vector<std::vector<EIndex>>
	getConnectivityFaceEIndices(const UMesh<freal,NDIM>& lm,
	                            const std::vector<bool>& isPhyBounPoint) const;

	/// The global mesh to partition
	const UMesh<freal,NDIM>& gm;

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
	TrivialReplicatedGlobalMeshPartitioner(const UMesh<freal,NDIM>& global_mesh);

	void compute_partition();
};

/// A simple partitioner that just serially partitions the mesh using Scotch
class ScotchRGMPartitioner : public ReplicatedGlobalMeshPartitioner
{
public:
	ScotchRGMPartitioner(const UMesh<freal,NDIM>& global_mesh);

	void compute_partition();
};

}

#endif

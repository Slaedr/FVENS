/** \file
 * \brief Implementation of mesh partitioning - calls PT-Scotch.
 */

#include <iostream>
#include <map>
#include "meshpartitioning.hpp"
#include "utilities/mpiutils.hpp"

namespace fvens {

ReplicatedGlobalMeshPartitioner::ReplicatedGlobalMeshPartitioner(const UMesh2dh<a_real>& globalmesh)
	: gm{global_mesh}
{ }

UMesh2dh<a_real> ReplicatedGlobalMeshPartitioner::restrictMeshToPartitions() const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	// const int nranks = get_mpi_size(MPI_COMM_WORLD);

	UMesh2dh<a_real> lm;
	lm.nelem = 0;
	for(a_int iel = 0; iel < gm.nelem; iel++)
		if(elemdist[iel] == rank)
			lm.nelem++;

	lm.maxnnode = gm.maxnnode;
	lm.maxnfael = gm.maxnfael;
	lm.nnofa = gm.nnofa;

	//! 1. Copy inpoel, get local to global elem map
	lm.inpoel.resize(lm.nelem, gm.maxnnode);
	lm.nfael.resize(lm.nelem);
	lm.nnode.resize(lm.nelem);
	lm.nbtag = gm.nbtag;
	lm.ndtag = gm.ndtag;
	assert(gm.ndtag == gm.vol_regions.cols());
	lm.vol_regions.resize(lm.nelem, gm.vol_regions.cols());
	const std::vector<a_int> elemLoc2Glob = extractInpoel(lm);

	//! 2. Copy required point coords into local mesh; get global to local point map
	const std::map<a_int,a_int> pointGlob2Loc = extractPointCoords(lm);

	//! 3. Convert inpoel entries from global indices to local
	for(a_int iel = 0; iel < lm.nelem; iel++)
		for(int j = 0; j < lm.nnode[iel]; j++)
			lm.inpoel(iel,j) = pointGlob2Loc[lm.inpoel(iel,j)];

	//! 4. Use point global-to-local map to identify global bfaces that are needed in this rank.
	//!    Copy them. In 3D this will be N^(2/3) log n. (N is global size, n is local size)
	lm.nface = 0;
	std::vector<std::array<a_int,4>> tbfaces;         // assuming max 4 points per face
	assert(gm.nnofa < 4);
	for(a_int iface = 0; iface < gm.nface; iface++)
	{
		bool reqd = true;
		std::array<a_int,4> locbfpoints;
		for(int j = 0; j < gm.nnofa; j++)
		{
			a_int locpointind = -1;
			try {
				locbfpoints[j] = pointGlob2Loc.at(gm.bface(iface,j));
			} catch (const std::out_of_range& oor) {
				reqd = false;
				break;
			}
		}
		if(reqd) {
			tbfaces.push_back(locbfpoints);
			lm.nface++;
		}
	}

	tbfaces.shrink_to_fit();
	lm.bface.resize(lm.nface, lm.nnofa+lm.nbtag);

	// 5. Compute global and local esuel. Also compute the intfac and elemface for the local mesh.
	//    Use these and local-to-global elem map to build the connect face structure.

	return lm;
}

std::vector<a_int>
ReplicatedGlobalMeshPartitioner::extractInpoel(UMesh2dh<a_real>& lm) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	std::vector<a_int> elemLoc2Glob(lm.nelem);

	a_int lociel = 0;
	for(a_int iel = 0; iel < gm.nelem; iel++)
		if(elemdist[iel] == rank)
		{
			elemLoc2Glob[lociel] = iel;
			for(int j = 0; j < gm.nnode[iel]; j++)
				lm.inpoel(lociel,j) = gm.inpoel(iel,j);
			for(int j = 0; j < gm.vol_regions.cols(); j++)
				lm.vol_regions(lociel,j) = gm.vol_regions(iel,j);
			lm.nnode[lociel] = gm.nnode[iel];
			lm.nfael[lociel] = gm.nfael[iel];
			lociel++;
		}
	assert(lociel == lm.nelem);
	return elemLoc2Glob;
}

std::map<a_int,a_int> ReplicatedGlobalMeshPartitioner::extractPointCoords(UMesh2dh<a_real>& lm) const
{
	//const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	// get global indices of the points needed on this rank
	std::vector<a_int> locpoints;
	locpoints.reserve(2*gm.npoin/nranks);
	for(a_int iel = 0; iel < lm.inpoel.rows(); iel++)
		for(int inode = 0; inode < lm.nnode[iel]; inode++)
			locpoints.push_back(lm.inpoel(iel,inode));

	// sort and remove duplicates
	std::sort(locpoints.begin(), locpoints.end());
	auto endpoint = std::unique(locpoints.begin(), locpoints.end());
	locpoints.erase(endpoint, locpoints.end());

	lm.npoin = static_cast<a_int>(locpoints.size());
	lm.coords.resize(lm.npoin,NDIM);

	// Get the global-to-local point index map - this has n log n cost.
	std::map<a_int,a_int> glbToLocPointMap;
	for(a_int i = 0; i < lm.npoin; i++)
		glbToLocPointMap[locpoints[i]] = i;

	a_int globpointer = 0, locpointer = 0;
	while(globpointer < gm.npoin && locpointer < lm.npoin)
	{
		if(globpointer == locpoints[locpointer])
		{
			for(int i = 0; i < NDIM; i++)
				lm.coords(locpointer,i) = gm.coords(globpointer,i);
			locpointer++;
		}
		globpointer++;
	}

	return glbToLocPointMap;
}

TrivialReplicatedGlobalMeshPartitioner::
TrivialReplicatedGlobalMeshPartitioner(const UMesh2dh<a_real>& globalmesh)
	: ReplicatedGlobalMeshPartitioner(global_mesh)
{ }

void TrivialReplicatedGlobalMeshPartitioner::compute_partition()
{
}

/// Computes the number of elements on each process
/** \param gm The global mesh
 * \param glbElemDist The MPI rank to which each element in the global mesh goes - assumed to be
 *   accessible only on rank 0.
 * \return A vector of number of elements. On rank zero, it is of length nprocs. On all other ranks,
 *   it has length 1 and contains the number of elements on that rank only.
 */
static std::vector<a_int> getNumLocalElems(const MeshData& gm,
                                           const std::vector<a_int>& glbElemDist)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	std::vector<a_int> nlocelem;

	if(rank == 0) {
		nlocelem.resize(nranks);
		std::fill(nlocelem.begin(), nlocelem.end(), 0);
		assert(static_cast<a_int>(glbElemDist.size()) == gm.nelem);

		for(a_int iel = 0; iel < gm.nelem; iel++)
			nlocelem[glbElemDist[iel]]++;
	}
	else {
		nlocelem.resize(1);
	}

	a_int mylocelem;
	MPI_Scatter(&nlocelem[0], 1, FVENS_MPI_INT, &mylocelem, 1, FVENS_MPI_INT, 0, MPI_COMM_WORLD);
	if(rank != 0)
		nlocelem[0] = mylocelem;
	else
		assert(nlocelem[0] == mylocelem);

#ifdef DEBUG
	if(rank == 0) {
		std::cout << " Number of elems on each rank:\n";
		for(int i = 0; i < nranks; i++)
			std::cout << "  Rank " << i << ": " << nlocelem[i] << '\n';
		std::cout << std::endl;
	} else {
		std::cout << " Number of elems on rank " << rank << ": " << nlocelem[0] << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	return nlocelem;
}

/// Get local elem-node connectivity matrix, number of nodes per elem and number of faces per elem,
///  and mapping from local to global element numbering from root process to respective processes
static void getElemDataFromRoot(const MeshData& gm, const std::vector<a_int>& numlocalelems,
                                const std::vector<int>& glbElemDist,
                                amat::Array2d<a_int>& inpoel, std::vector<int>& nnode,
                                std::vector<int>& nfael, std::vector<int>& loc2globElem)
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	// transfer element data from root to each rank
	if(rank == 0) {
		for(int irnk = 0; irnk < nranks; irnk++)
		{
			amat::Array2d<a_int> inpoelbuffer(numlocalelems[irnk],gm.maxnnode);
			std::vector<int> nnodebuffer(numlocalelems[irnk]), nfaelbuffer(numlocalelems[irnk]);
			std::vector <int> loc2globElembuffer(numlocalelems[irnk]);

			a_int lociel = 0;
			for(a_int iel = 0; iel < gm.nelem; iel++) {
				if(glbElemDist[iel] == irnk) {
					for(int inode = 0; inode < gm.maxnnode; inode++)
						inpoelbuffer(lociel,inode) = gm.inpoel(iel,inode);
					loc2globElembuffer[lociel] = iel;
					nnodebuffer[lociel] = gm.nnode[iel];
					nfaelbuffer[lociel] = gm.nfael[iel];
					lociel++;
				}
			}

			assert(lociel == localelemsizes[irnk]);

			MPI_Request req[4];
			MPI_Isend(&inpoelbuffer(0,0), lociel*gm.maxnnode, FVENS_MPI_INT, irnk, 0, MPI_COMM_WORLD,
			          &req[0]);
			MPI_Isend(&nnodebuffer[0], lociel, FVENS_MPI_INT, irnk, 1, MPI_COMM_WORLD, &req[1]);
			MPI_Isend(&nfaelbuffer[0], lociel, FVENS_MPI_INT, irnk, 2, MPI_COMM_WORLD, &req[2]);
			MPI_Isend(&loc2globElembuffer[0], lociel, FVENS_MPI_INT, irnk, 3, MPI_COMM_WORLD,
			          &req[3]);
			MPI_Status statuses[4];
			int ierr = MPI_Waitall(4, req, statuses);
			assert(ierr == MPI_SUCCESS);
			(void)ierr;
		}
	}

	MPI_Request req[4];
	MPI_Irecv(&inpoel(0,0), numlocalelems[0]*gm.maxnnode, FVENS_MPI_INT, 0, 0,
	          MPI_COMM_WORLD, &req[0]);
	MPI_Irecv(&nnode[0], numlocalelems[0], FVENS_MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
	MPI_Irecv(&nfael[0], numlocalelems[0], FVENS_MPI_INT, 0, 2, MPI_COMM_WORLD, &req[2]);
	MPI_Irecv(&loc2globElem[0], numlocalelems[0], FVENS_MPI_INT, 0, 3, MPI_COMM_WORLD, &req[3]);
	MPI_Status statuses[4];
	int ierr = MPI_Waitall(4, req, statuses);
	assert(ierr == MPI_SUCCESS);
	(void)ierr;
}

/// Determines what points are needed for each rank and transfers the required points from rank 0
/** \param[in] gm The global mesh
 * \param[out] lnpoin Number of local points requried for this rank.
 * \param[out] lcoords Coordinates of the points requried for this rank.
 * \return A map from global indices corresponding to points present in the calling rank to
 * the local indices.
 */
static std::map<a_int,a_int> splitPointData(const MeshData& gm,
                                            const amat::Array2d<a_int>& linpoel,
                                            const std::vector<int>& lnnode,
                                            int& lnpoin, amat::Array2d<a_real>& lcoords)
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	// get global indices of the points needed on this rank
	std::vector<a_int> locpoints;
	locpoints.reserve(2*gm.npoin/nranks);
	for(a_int iel = 0; iel < linpoel.rows(); iel++)
		for(int inode = 0; inode < lnnode[iel]; inode++)
			locpoints.push_back(linpoel(iel,inode));

	// sort and remove duplicates
	std::sort(locpoints.begin(), locpoints.end());
	auto endpoint = std::unique(locpoints.begin(), locpoints.end());
	locpoints.erase(endpoint, locpoints.end());

	lnpoin = static_cast<a_int>(locpoints.size());
	lcoords.resize(lnpoin,NDIM);

	// Get the global-to-local point index map
	std::map<a_int,a_int> glbToLocPointMap;
	for(a_int i = 0; i < lnpoin; i++)
		glbToLocPointMap[locpoints[i]] = i;

	// Get number of local points for each rank into rank 0
	std::vector<size_t> locpointsizes;
	if(rank == 0)
		locpointsizes.resize(nranks);
	size_t locpointsz = locpoints.size();
	MPI_Gather(&locpointsz, 1, MPI_UNSIGNED_LONG_LONG, &locpointsizes[0], 1, MPI_UNSIGNED_LONG_LONG,
	           0, MPI_COMM_WORLD);

	// Transfer the actual point indices to root process
	if(rank != 0)
		MPI_Send(&locpoints[0], locpoints.size(), FVENS_MPI_INT, 0, 0, MPI_COMM_WORLD);

	if(rank == 0)
	{
		for(int irnk = 0; irnk < nranks; irnk++)
		{
			// Get the point indices requried for this rank
			std::vector<a_int> pointinds(locpointsizes[irnk]);
			if(irnk != 0)
				MPI_Recv(&pointinds[0], locpointsizes[irnk], FVENS_MPI_INT, irnk, 0, MPI_COMM_WORLD,
			         MPI_STATUS_IGNORE);
			else
				std::copy(locpoints.begin(), locpoints.end(), pointinds.begin());

			amat::Array2d<a_real> loccoords(locpointsizes[irnk],NDIM);
			a_int globpointer = 0, locpointer = 0;
			while(globpointer < gm.npoin && locpointer < static_cast<a_int>(locpointsizes[irnk]))
			{
				if(globpointer == pointinds[locpointer])
				{
					for(int i = 0; i < NDIM; i++)
						loccoords(locpointer,i) = gm.coords(globpointer,i);
					locpointer++;
				}
				globpointer++;
			}

			if(locpointer != static_cast<a_int>(locpointsizes[irnk]))
				throw std::logic_error("Could not account for all nodes!");

			if(irnk != 0)
				MPI_Send(&loccoords(0,0), locpointsizes[irnk]*NDIM, FVENS_MPI_REAL, irnk, 1,
				          MPI_COMM_WORLD);
			else
				for(a_int ip = 0; ip < static_cast<a_int>(locpointsizes[0]); ip++)
					for(int j = 0; j < NDIM; j++)
						lcoords(ip,j) = loccoords(ip,j);
		}
	}

	MPI_Status status;
	MPI_Recv(&lcoords(0,0), lnpoin*NDIM, FVENS_MPI_REAL, 0, 1, MPI_COMM_WORLD, &status);
	assert(status == MPI_SUCCESS);

	return glbToLocPointMap;
}

void splitMeshArrays(const MeshData& gm,
                     const std::vector<int>& glbElemDist,
                     UMesh2dh<a_real>& lm)
{
	// const int nranks = get_mpi_size(MPI_COMM_WORLD);
	// const int rank = get_mpi_rank(MPI_COMM_WORLD);

	const std::vector<a_int> numlocelems = getNumLocalElems(gm, glbElemDist);

	// Get local sizes from root

	lm.nelem = numlocelems[0];
	lm.inpoel.resize(numlocelems[0], gm.maxnnode);
	lm.nnode.resize(numlocelems[0]);
	lm.nfael.resize(numlocelems[0]);
	std::vector<a_int> loc2globElem(numlocelems[0]); // mapping from local elem index to global elem index

	getElemDataFromRoot(gm, numlocelems, glbElemDist, lm.inpoel, lm.nnode, lm.nfael, loc2globElem);

	std::map<a_int,a_int> glob2locPoint = splitPointData(gm, lm.inpoel, lm.nnode, lm.npoin, lm.coords);

	// use the point mapping to localize the inpoel arrays of local meshes
	//  This could be an expensive process because each lookup of glob2locPoint is log(lm.npoin).
	for(a_int iel = 0; iel < lm.nelem; iel++)
		for(int i = 0; i < lm.nnode[iel]; i++)
			lm.inpoel(iel,i) = glob2locPoint[lm.inpoel(iel,i)];

#ifdef DEBUG
	std::cout << " Rank " << rank << ": Number of elems = " << lm.nelem
	          << ", number of nodes = " << lm.npoin << std::endl;
#endif
}

UMesh2dh<a_real> partitionMeshTrivial(const MeshData& gm)
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	UMesh2dh<a_real> lm;  // local mesh

	const int numloceleminit = gm.nelem / nranks;
	const int numlocalelemremain = gm.nelem % nranks;
	bool err = false;
	if(rank != nranks)
		lm.nelem = numloceleminit;
	else {
		lm.nelem = numloceleminit + numlocalelemremain;
		if(lm.nelem != gm.nelem - (nranks-1)*numloceleminit)
			err = true;
	}
	if(err)
		throw std::logic_error("Trivial partition does not add up!");

	std::vector<int> glbElemDist(1,0);
	if(rank == 0)
	{
		glbElemDist.resize(gm.nelem);
		for(int irank = 0; irank < nranks; irank++) {
			for(a_int iel = irank*numloceleminit; iel < (irank+1)*numloceleminit; iel++)
				glbElemDist[iel] = irank;
		}
		for(a_int iel = nranks*numloceleminit; iel < gm.nelem; iel++)
			glbElemDist[iel] = nranks-1;
	}

	splitMeshArrays(gm, glbElemDist, lm);

	return lm;
}

}

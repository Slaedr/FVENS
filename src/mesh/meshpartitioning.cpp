/** \file
 * \brief Implementation of mesh partitioning - calls PT-Scotch.
 */

#include <iostream>
#include <map>
#include <scotch.h>
#include "meshpartitioning.hpp"
#include "utilities/mpiutils.hpp"

namespace fvens {

ReplicatedGlobalMeshPartitioner::ReplicatedGlobalMeshPartitioner(const UMesh<freal,2>& globalmesh)
	: gm{globalmesh}
{
	const int mpisize = get_mpi_size(MPI_COMM_WORLD);
	if(gm.gnelem() < mpisize)
		throw std::runtime_error("Not enough cells in this mesh for " +
		                         std::to_string(mpisize) + " processes!");
}

UMesh<freal,2> ReplicatedGlobalMeshPartitioner::restrictMeshToPartitions() const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	UMesh<freal,2> lm;
	lm.nelem = 0;
	for(fint iel = 0; iel < gm.nelem; iel++)
		if(elemdist[iel] == rank)
			lm.nelem++;

	lm.maxnnode = gm.maxnnode;
	lm.maxnfael = gm.maxnfael;
	lm.nnofa = gm.nnofa;
	lm.nelemglobal = gm.nelem;
	lm.npoinglobal = gm.npoin;
#ifdef DEBUG
	std::cout << "ReplicatedGlobalMeshPartitioner: Rank " << rank << ": Nelem = " << lm.nelem
	          << std::endl;
#endif

	//! 1. Copy inpoel, get local to global elem map
	lm.inpoel.resize(lm.nelem, gm.maxnnode);
	lm.nfael.resize(lm.nelem);
	lm.nnode.resize(lm.nelem);
	lm.nbtag = gm.nbtag;
	lm.ndtag = gm.ndtag;
	assert(gm.ndtag == gm.vol_regions.cols());
	lm.vol_regions.resize(lm.nelem, gm.vol_regions.cols());

	lm.globalElemIndex = extractInpoel(lm);

	/*! 2. Copy required point coords into local mesh
	 *     get global to local and local-to-global point maps
	 */
	std::vector<fint> pointLoc2Glob;
	const std::map<fint,fint> pointGlob2Loc = extractPointCoords(lm, pointLoc2Glob);

#ifdef DEBUG
	for(fint ip = 0; ip < lm.npoin; ip++)
		assert(pointGlob2Loc.at(pointLoc2Glob[ip]) == ip);
#endif

	//! 3. Convert inpoel entries from global indices to local
	for(fint iel = 0; iel < lm.nelem; iel++)
		for(int j = 0; j < lm.nnode[iel]; j++)
			lm.inpoel(iel,j) = pointGlob2Loc.at(lm.inpoel(iel,j));

	/*! 4. Use point global-to-local map to identify global bfaces that are needed in this rank.
	 *     Copy them. In 3D this will be N^(2/3) log n. (N is global size, n is local size)
	 */
	extractbfaces(pointGlob2Loc, lm);

	/*! 5. Compute local esuel and mark elements that neighbor and points that lie on
	 *     a physical boundary face
	 */
	lm.compute_elementsSurroundingPoints();
	lm.compute_elementsSurroundingElements();
	const std::vector<bool> isPhyBounPoint = markLocalPhysicalBoundaryPoints(lm);
	assert(isPhyBounPoint.size() == static_cast<size_t>(lm.npoin));

	/*! 6. Use local esuel, the global esuel and local-to-global elem map 
	 *     to build the connectivity face structure. Use the fact that ordering of faces within
	 *     elements is not changed during restriction to assign global face indices.
	 */
	const std::vector<std::vector<EIndex>> connElemLocalFace
		= getConnectivityFaceEIndices(lm, isPhyBounPoint);

	lm.nconnface = 0;
	for(fint i = 0; i < lm.nelem; i++)
		lm.nconnface += static_cast<fint>(connElemLocalFace[i].size());

	if(lm.nconnface > 0)
		lm.connface.resize(lm.nconnface,5);
	fint icofa = 0;
	for(fint iel = 0; iel < lm.nelem; iel++)
	{
		for(size_t iconface = 0; iconface < connElemLocalFace[iel].size(); iconface++)
		{
			const EIndex localConnFace = connElemLocalFace[iel][iconface];
			lm.connface(icofa,0) = iel;
			lm.connface(icofa,1) = localConnFace;
			lm.connface(icofa,2) = -1;
			lm.connface(icofa,3) = -1;
			lm.connface(icofa,4) = gm.gelemface(lm.globalElemIndex[iel],localConnFace);

			std::vector<fint> locfacepoints(lm.nnofa,-1);  // points of the connectivity face
			for(FIndex linofa = 0; linofa < lm.nnofa; linofa++)
				locfacepoints[linofa] = lm.inpoel(iel, lm.getNodeEIndex(iel,localConnFace,linofa));

			const fint glind = lm.globalElemIndex[iel];

			// identify the face of the global element which matches the index of connectivity face
			for(EIndex jgf = 0; jgf < gm.nfael[glind]; jgf++)
			{
				bool matched = true;
				for(FIndex jnofa = 0; jnofa < gm.nnofa; jnofa++)
				{
					const fint globpoint = gm.inpoel(glind, gm.getNodeEIndex(glind,jgf,jnofa));
					bool pointmatched = false;
					for(FIndex linofa = 0; linofa < lm.nnofa; linofa++)
					{
						if(pointLoc2Glob[locfacepoints[linofa]] == globpoint)
						{
							// this point (jnofa) is matched, now check the next one
							pointmatched = true;
							break;
						}
					}
					if(!pointmatched) {
						// the global point corresponding to jnofa did not match any point in the
						//  local face 'localConnFace', so the the face jgf is not the one we need.
						//  Go the next face of element glind.
						matched = false;
						break;
					}
				}

				if(matched) {
					// both points of face jgf match local face 'localConnFace'
					lm.connface(icofa,2) = elemdist[gm.esuel(glind,jgf)];
					lm.connface(icofa,3) = gm.esuel(glind,jgf);
					break;
				}
			}

			if(lm.connface(icofa,2) < 0)
				throw std::logic_error("Could not find connectivity face!");

			icofa++;
		}
	}

	assert(icofa == lm.nconnface);

	return lm;
}

std::vector<fint>
ReplicatedGlobalMeshPartitioner::extractInpoel(UMesh<freal,2>& lm) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	std::vector<fint> elemLoc2Glob(lm.nelem);

	fint lociel = 0;
	for(fint iel = 0; iel < gm.nelem; iel++)
		if(elemdist[iel] == rank)
		{
			elemLoc2Glob[lociel] = iel;
			for(int j = 0; j < gm.maxnnode; j++)
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

std::map<fint,fint>
ReplicatedGlobalMeshPartitioner::
extractPointCoords(UMesh<freal,2>& lm, std::vector<fint>& locpoints) const
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	// get global indices of the points needed on this rank
	locpoints.reserve(2*gm.npoin/nranks);
	for(fint iel = 0; iel < lm.inpoel.rows(); iel++)
		for(int inode = 0; inode < lm.nnode[iel]; inode++)
			locpoints.push_back(lm.inpoel(iel,inode));

	// sort and remove duplicates
	std::sort(locpoints.begin(), locpoints.end());
	auto endpoint = std::unique(locpoints.begin(), locpoints.end());
	locpoints.erase(endpoint, locpoints.end());

	lm.npoin = static_cast<fint>(locpoints.size());
	lm.coords.resize(lm.npoin,NDIM);

	// Get the global-to-local point index map - this has n log n cost.
	std::map<fint,fint> glbToLocPointMap;
	for(fint i = 0; i < lm.npoin; i++) {
		glbToLocPointMap[locpoints[i]] = i;
	}

	fint globpointer = 0, locpointer = 0;
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

void ReplicatedGlobalMeshPartitioner::extractbfaces(const std::map<fint,fint>& pointGlob2Loc,
                                                    UMesh<freal,2>& lm) const
{
	const int irank = get_mpi_rank(MPI_COMM_WORLD);

	lm.nbface = 0;
	std::vector<std::array<fint,6>> tbfaces;         // assuming max 4 points and 2 tags per face
	assert(gm.nnofa <= 4);
	assert(gm.nbtag <= 2);
	for(fint iface = 0; iface < gm.nbface; iface++)
	{
		bool reqd = true;
		std::array<fint,6> locbfpoints;
		// for(int j = 0; j < gm.nnofa; j++)
		// {
		// 	try {
		// 		locbfpoints[j] = pointGlob2Loc.at(gm.bface(iface,j));
		// 	} catch (const std::out_of_range& oor) {
		// 		reqd = false;
		// 		break;
		// 	}
		// }
		const fint globelem = gm.gintfac(iface+gm.gPhyBFaceStart(),0);
		if(elemdist[globelem] == irank)
		{
			for(int j = 0; j < gm.nnofa; j++)
				locbfpoints[j] = pointGlob2Loc.at(gm.bface(iface,j));
		}
		else
			reqd = false;

		if(reqd) {
			for(int j = 0; j < gm.nbtag; j++)
				locbfpoints[gm.nnofa+j] = gm.bface(iface,gm.nnofa+j);
			tbfaces.push_back(locbfpoints);
			lm.nbface++;
		}

#ifdef DEBUG
		// Do the global intfac and element partition agree that this face is in this partition?
		if(reqd) {
			const fint globelem = gm.gintfac(iface+gm.gPhyBFaceStart(),0);
			assert(elemdist[globelem] == irank);
		}
#endif
	}

	lm.bface.resize(lm.nbface, lm.nnofa+lm.nbtag);
	for(fint iface = 0; iface < lm.nbface; iface++)
		for(int j = 0; j < lm.nnofa+lm.nbtag; j++)
			lm.bface(iface,j) = tbfaces[iface][j];
}

std::vector<bool>
ReplicatedGlobalMeshPartitioner::
markLocalPhysicalBoundaryPoints(const UMesh<freal,2>& lm) const
{
	std::vector<bool> isBounPoin(lm.npoin,false);

	for(fint iface = 0; iface < lm.nbface; iface++)
	{
		for(int inode = 0; inode < lm.nnofa; inode++) {
			isBounPoin[lm.bface(iface,inode)] = true;
		}
	}
	return isBounPoin;
}

std::vector<std::vector<EIndex>>
ReplicatedGlobalMeshPartitioner::
getConnectivityFaceEIndices(const UMesh<freal,2>& lm,
                            const std::vector<bool>& isPhyBounPoint) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	// Non-negative if an element contains at least one connectivity face, in which case
	//  it stores the connectivity faces' index w.r.t. the element (their EIndex) in this subdomain
	std::vector<std::vector<EIndex>> connElemLocalFace(lm.nelem);

	for(fint iel = 0; iel < lm.nelem; iel++)
	{
		// Check whether face points are physical boundary points to identify connectivity faces
		for(EIndex iface = 0; iface < lm.nfael[iel]; iface++)
		{
			if(lm.esuel(iel,iface) == -1)
			{
				bool isconnface = false;
				/* Get subdomain index of the points on this face.
				 * If at least one point is not on a physical boundary, this is a connectivity face.
				 */
				for(FIndex inode = 0; inode < lm.nnofa; inode++)
				{
					const fint locpoint = lm.inpoel(iel,lm.getNodeEIndex(iel,iface,inode));
					if(!isPhyBounPoint[locpoint]) {
						isconnface = true;
						break;
					}
				}
				if(isconnface) {
					connElemLocalFace[iel].push_back(iface);
				}
			}
		}
	}
	std::cout << "Rank " << rank << ": computed connElemLocalFace" << std::endl;
	return connElemLocalFace;
}

bool ReplicatedGlobalMeshPartitioner::checkConnFaces(const UMesh<freal,2>& lm) const
{
	bool match = true;
	for(fint iface = lm.gConnBFaceStart(); iface < lm.gConnBFaceEnd(); iface++)
	{
		const fint leftelem = lm.gglobalElemIndex(lm.gintfac(iface,0));
		const fint globface = lm.gconnface(iface-lm.gConnBFaceStart(),4);
		assert(globface < gm.gnaface());
		if(leftelem != gm.gintfac(globface,0)) {
			match = false;
			break;
		}
	}
	return match;
}

TrivialReplicatedGlobalMeshPartitioner::
TrivialReplicatedGlobalMeshPartitioner(const UMesh<freal,2>& globalmesh)
	: ReplicatedGlobalMeshPartitioner(globalmesh)
{ }

void TrivialReplicatedGlobalMeshPartitioner::compute_partition()
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int numloceleminit = gm.gnelem() / nranks;
	//const int numlocalelemremain = gm.gnelem() % nranks;
	elemdist.resize(gm.gnelem());

	for(int irank = 0; irank < nranks; irank++) {
		for(fint iel = irank*numloceleminit; iel < (irank+1)*numloceleminit; iel++)
			elemdist[iel] = irank;
	}
	for(fint iel = nranks*numloceleminit; iel < gm.gnelem(); iel++)
		elemdist[iel] = nranks-1;
}

ScotchRGMPartitioner::ScotchRGMPartitioner(const UMesh<freal,2>& globalmesh)
	: ReplicatedGlobalMeshPartitioner(globalmesh)
{ }

void ScotchRGMPartitioner::compute_partition()
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0) {
		SCOTCH_Graph *sgraph = SCOTCH_graphAlloc();

		elemdist.resize(gm.gnelem());

		SCOTCH_memFree(sgraph);
	}
}

SimpleRGMPartitioner::SimpleRGMPartitioner(const UMesh<freal,2>& globalmesh)
	: ReplicatedGlobalMeshPartitioner(globalmesh)
{ }

void SimpleRGMPartitioner::compute_partition()
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int numloceleminit = gm.gnelem() / nranks;
	elemdist.resize(gm.gnelem());

	std::vector<bool> ismarked(gm.gnelem(), false);

	fint startcell = 0;

	for(int irank = 0; irank < nranks; irank++)
	{
		const fint ncellsrank = (irank == nranks-1) ?
			(gm.gnelem()-numloceleminit*(nranks-1)) : numloceleminit;

		bool endoftheline = false;        // True if no unmarked neighbors exist
		fint nmarkedcells = 0;           // Number of cells marked

		std::vector<fint> cellsinrank;   // Will contain this rank's cells
		cellsinrank.reserve(ncellsrank);

		fint baseidx = 0;                // Cell in cellsinrank from which to mark neighbors
		ismarked[startcell] = true;
		cellsinrank.push_back(startcell);
		nmarkedcells++;

		while(!endoftheline && nmarkedcells <= ncellsrank)
		{
			const fint basecell = cellsinrank[baseidx];
			for(int iface = 0; iface < gm.gnfael(basecell); iface++)
			{
				if(nmarkedcells >= ncellsrank)
					break;

				const fint nbdcell = gm.gesuel(startcell,iface);
				if(!ismarked[nbdcell] && nbdcell < gm.gnelem())
				{
					cellsinrank.push_back(nbdcell);
					nmarkedcells++;
					ismarked[nbdcell] = true;
				}
			}

			baseidx++;
		}
	}
}

}

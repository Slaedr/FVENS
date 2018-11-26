/** \file
 * \brief Implementation of mesh partitioning - calls PT-Scotch.
 */

#include <iostream>
#include "meshpartitioning.hpp"
#include "utilities/mpiutils.hpp"

namespace fvens {

static a_int getNumLocalElems(const int rank, const MeshData& gm,
                              const a_int *const glbElemDist)
{
	a_int nlocelem = 0;
	//#pragma omp parallel for default(shared) reduction(+:nlocelem)
	for(a_int iel = 0; iel < gm.nelem; iel++) {
		if(glbElemDist[iel] == rank)
			nlocelem++;
	}
	return nlocelem;
}

/// Get local elem-node connectivity matrix, number of nodes per elem and number of faces per elem,
///  and mapping from local to global element numbering from root process to respective processes
static void getDataFromRoot(const MeshData& gm, const a_int nlocelem,
                            amat::Array2d<a_int>& inpoel, std::vector<int>& nnode,
                            std::vector<int>& nfael, std::vector<int>& loc2globElem)
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	// transfer local sizes from each rank to root
	std::vector<a_int> localelemsizes;
	if(rank == 0) localelemsizes.resize(nranks);
	MPI_Gather((void*)&nlocelem, 1, FVENS_MPI_INT, (void*)&localelemsizes[0],1,FVENS_MPI_INT, 0,
	           MPI_COMM_WORLD);

	// transfer element data from root to each rank
	if(rank == 0) {
		for(int irnk = 0; irnk < nranks; irnk++) {
			amat::Array2d<a_int> inpoelbuffer(localelemsizes[irnk],gm.maxnnode);
			std::vector<int> nnodebuffer(localelemsizes[irnk]), nfaelbuffer(localelemsizes[irnk]);
			std::vector <int> loc2globElembuffer(localelemsizes[irnk]);

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
			MPI_Isend(&inpoelbuffer[0], lociel*gm.maxnnode, FVENS_MPI_INT, irnk, 0, MPI_COMM_WORLD,
			          &req[0]);
			MPI_Isend(&nnodebuffer[0], lociel, FVENS_MPI_INT, irnk, 1, MPI_COMM_WORLD, &req[1]);
			MPI_Isend(&nfaelbuffer[0], lociel, FVENS_MPI_INT, irnk, 2, MPI_COMM_WORLD, &req[2]);
			MPI_Isend(&loc2globElembuffer[0], lociel, FVENS_MPI_INT, irnk, 3, MPI_COMM_WORLD,
			          &req[3]);
			MPI_Status statuses[4];
			int ierr = MPI_Waitall(4, req, statuses);
			assert(ierr == MPI_SUCCESS);
		}
	}

	MPI_Request req[4];
	MPI_Irecv(&inpoel(0,0), localelemsizes[rank]*gm.maxnnode, FVENS_MPI_INT, 0, 0,
	          MPI_COMM_WORLD, &req[0]);
	MPI_Irecv(&nnode[0], localelemsizes[rank], FVENS_MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
	MPI_Irecv(&nfael[0], lociel, FVENS_MPI_INT, 0, 2, MPI_COMM_WORLD, &req[2]);
	MPI_Irecv(&loc2globElem[0], lociel, FVENS_MPI_INT, 0, 3, MPI_COMM_WORLD, &req[3]);
	MPI_Status statuses[4];
	int ierr = MPI_Waitall(4, req, statuses);
	assert(ierr == MPI_SUCCESS);
	(void)ierr;
}

/// Populates this process's share of mesh arrays from the global arrays
/** Assumptions: gm's integers and the array glbElemDist are available on all ranks.
 * gm's arrays are only available on rank 0.
 */
static void splitMeshArrays(const MeshData& gm,
                            const a_int *const glbElemDist,
                            UMesh2dh<a_real>& lm)
{
	const int nranks = get_mpi_size(MPI_COMM_WORLD);
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	const a_int nlocelem = getNumLocalElems(rank, gm, glbElemDist);

	lm.nelem = nlocelem;
	lm.inpoel.resize(nlocelem, gm.maxnnode);
	lm.nnode.resize(nlocelem);
	lm.nfael.resize(nlocelem);
	std::vector<a_int> loc2globElem(nlocelem);  // mapping from local elem index to global elem index

	getDataFromRoot(gm, nlocelem, lm.inpoel, lm.nnode, lm.nfael, loc2globElem);

	std::vector<int> locpoints(gm.npoin,0);
	for(a_int iel = 0; iel < lm.nelem; iel++)
		for(int inode = 0; inode < lm.nnode[iel]; inode++)
			locpoints[lm.inpoel(iel,inode)] = 1;
	a_int nlocpoin=0;
	for(a_int ip = 0; ip < gm.npoin; ip++)
		nlocpoin += locpoints[ip];
	lm.npoin = nlocpoin;
	lm.coords.reize(nlocpoin);

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

	const std::vector<a_int> glbElemDist(gm.nelem,0);
	for(int irank = 0; irank < nranks; irank++) {
		for(a_int iel = irank*numloceleminit; iel < (irank+1)*numloceleminit; iel++)
			glbElemDist[iel] = irank;
	}
	for(a_int iel = nranks*numloceleminit; iel < gm.nelem; iel++)
		glbElemDist[iel] = nranks-1;

	splitMeshArrays(gm, &glbElemDist[0], lm);

	return lm;
}

}

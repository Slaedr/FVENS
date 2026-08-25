/** \file
 * \brief Tests for distributed parallel mesh preprocessing
 */

#undef NDEBUG

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <petscsys.h>
#include "utilities/mpiutils.hpp"
#include "mesh/meshpartitioning.hpp"
#include "mesh/ameshutils.hpp"
#include "linalg/alinalg.hpp"
#include "linalg/petscutils.hpp"

using namespace fvens;

/// Gets global element numbering
std::vector<fint> getElemDist(const fint nelem, std::ifstream& fin)
{
	std::vector<fint> elems(nelem);
	std::string dum;
	fin >> dum;
	for(fint i = 0; i < nelem; i++)
		fin >> elems[i];
	return elems;
}

/// Gets connectivity face data
/// assumes getElemDist has been called using the same ifstream beforehand
amat::Array2d<fint> getConnMatrix(const fint nconnface, std::ifstream& fin)
{
	amat::Array2d<fint> conn;
	if(nconnface > 0)
		conn.resize(nconnface,4);
	std::string dum;
	fin >> dum;
	for(fint i = 0; i < nconnface; i++)
		for(int j = 0; j < 4; j++)
			fin >> conn(i,j);
	return conn;
}

// Checks a trivial distribution in which the cells are uniformly divided according to their index
//  in the mesh file
void checkTrivial(const std::string globalmeshfile, const std::vector<std::string>& localmeshfiles,
                  const std::vector<std::string>& distfiles)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	UMesh<freal,NDIM> gm(readMesh(globalmeshfile));
	gm.compute_topological();

	std::shared_ptr<ReplicatedGlobalMeshPartitioner> p;
	p = std::make_shared<TrivialReplicatedGlobalMeshPartitioner>(gm);

	p->compute_partition();
	const UMesh<freal,NDIM> lm = p->restrictMeshToPartitions();

	// Read solution to check against
	const UMesh<freal,NDIM> reflm(readMesh(localmeshfiles[rank]));
	std::ifstream fin(distfiles[rank]);
	if(!fin) {
		throw std::runtime_error("File not found!");
	}
	const std::vector<fint> elemglindices = getElemDist(lm.gnelem(), fin);
	const amat::Array2d<fint> connface = getConnMatrix(lm.gnConnFace(), fin);
	fin.close();

	// check
	const std::array<bool,8> isequal = compareMeshes(lm, reflm);

	for(int irnk = 0; irnk < nranks; irnk++)
	{
		MPI_Barrier(MPI_COMM_WORLD);

		if(rank == irnk) {
			std::cout << "Rank " << irnk << std::endl;
			assert(isequal[0]);
			assert(isequal[1]);
			assert(isequal[2]);
			assert(isequal[3]);
			assert(isequal[4]);
			assert(isequal[5]);
			assert(isequal[6]);
			assert(isequal[7]);

			for(fint i = 0; i < lm.gnelem(); i++) {
				assert(lm.gglobalElemIndex(i) == elemglindices[i]);
			}
			for(fint i = 0; i < lm.gnConnFace(); i++)
				for(int j = 0; j < 4; j++)
					assert(lm.gconnface(i,j) == connface(i,j));
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
}

/* For any supported partitioner, checks whether partitions are connected
 * That is, every cell of a partition should have at least 1 neighbour in the same partition. 
 */
void checkConnectedness(const UMesh<freal,NDIM>& gm, const std::string algo)
{

	std::shared_ptr<ReplicatedGlobalMeshPartitioner> p;
	if(algo == "scotch")
		p = std::make_shared<ScotchRGMPartitioner>(gm);
	else
		p = std::make_shared<TrivialReplicatedGlobalMeshPartitioner>(gm);

	p->compute_partition();

	UMesh<freal,NDIM> lm = p->restrictMeshToPartitions();
	lm.compute_topological();
	lm.compute_areas();
	lm.compute_face_data();
}

/* Checks, against an independent ground truth, that after mesh preprocessing (which may include
 * -mesh_reorder) this rank still correctly identifies:
 *  (a) which of its own current local cells owns each connectivity face (connface column 0,
 *      consumed via intfac/esuel for gradients, reconstruction, flux and Jacobian assembly), and
 *  (b) the current global cell index of each connectivity face's external neighbour cell
 *      (connface column 3 together with the globalElemIndex invariant
 *      globalElemIndex[i] == this rank's global cell offset + i), consumed by the PETSc
 *      ghost-vector machinery (createGhostedSystemVector / VecGhostUpdate).
 *
 * The ground truth is built from a *second*, never-reordered restriction of the same mesh
 * (same partitioner, so the same rank gets the same elements): there, connface column 0 and
 * globalElemIndex are correct by construction (see ReplicatedGlobalMeshPartitioner), so they
 * can be used to look up, from the raw global mesh, the true owner and neighbour cell centroids
 * for each connectivity face - keyed by the global face id (connface column 4), which is stable
 * across partitioning and reordering on either side.
 *
 * A ghost-vs-trace-vector comparison was deliberately NOT used here: both routes would end up
 * reading the *same* stale connface data (one directly, the other via intfac), so they keep
 * agreeing with each other even when both are wrong.
 */
void checkGhostConsistency(const std::string meshfile)
{
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	// --- Ground truth: an independent, never-reordered restriction of the same mesh. ---
	UMesh<freal,NDIM> ggm(readMesh(meshfile));
	ggm.compute_topological();
	std::vector<freal> ggcentres(ggm.gnelem()*NDIM);
	ggm.compute_cell_centres(&ggcentres[0]);

	TrivialReplicatedGlobalMeshPartitioner gp(ggm);
	gp.compute_partition();
	const UMesh<freal,NDIM> glm = gp.restrictMeshToPartitions();

	// For each connectivity face (keyed by its stable global face id): the true centroids of
	//  (this rank's owner cell, the external neighbour cell), in the original global mesh.
	std::map<fint, std::array<freal,2*NDIM>> truth;
	for(fint icface = 0; icface < glm.gnConnFace(); icface++)
	{
		std::array<freal,2*NDIM> c;
		const fint myGlobal = glm.gglobalElemIndex(glm.gconnface(icface,0));
		const fint nbGlobal = glm.gconnface(icface,3);
		for(int d = 0; d < NDIM; d++) {
			c[d]      = ggcentres[myGlobal*NDIM+d];
			c[NDIM+d] = ggcentres[nbGlobal*NDIM+d];
		}
		truth[glm.gconnface(icface,4)] = c;
	}

	// --- The mesh under test: goes through the real preprocessing path, including reordering. ---
	const UMesh<freal,NDIM> m = constructMesh(meshfile);

	// globalElemIndex must equal this rank's global cell offset plus local position, exactly.
	fint nelem = m.gnelem();
	fint offset = 0;
	const int mpiret = MPI_Exscan(&nelem, &offset, 1, FVENS_MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
	assert(mpiret == MPI_SUCCESS); (void)mpiret;
	if(mpirank == 0)
		offset = 0;
	for(fint i = 0; i < m.gnelem(); i++)
		assert(m.gglobalElemIndex(i) == offset + i);

	std::vector<freal> centres(m.gnelem()*NDIM);
	m.compute_cell_centres(&centres[0]);

	// (a) connface column 0: the cell it currently points to must be the true owner.
	for(fint icface = 0; icface < m.gnConnFace(); icface++)
	{
		const auto& tr = truth.at(m.gconnface(icface,4));
		const fint owner = m.gconnface(icface,0);
		assert(owner >= 0 && owner < m.gnelem());
		for(int d = 0; d < NDIM; d++)
			assert(std::fabs(centres[owner*NDIM+d] - tr[d]) < 1e-10);
	}

	// (b) Ghost route: fill a ghosted Vec with cell centroids, update the ghost layer, and check
	//     the fetched neighbour centroid against ground truth (exercises connface column 3 and
	//     the PETSc numbering).
	Vec gvec;
	int ierr = createGhostedSystemVector(&m, NVARS, &gvec);
	assert(ierr == 0);
	{
		MutableGhostedVecHandler<PetscScalar> gvh(gvec);
		PetscScalar *const gloc = gvh.getArray();
		for(fint i = 0; i < m.gnelem(); i++) {
			for(int d = 0; d < NDIM; d++)
				gloc[i*NVARS+d] = centres[i*NDIM+d];
			for(int d = NDIM; d < NVARS; d++)
				gloc[i*NVARS+d] = 0;
		}
	}
	ierr = VecGhostUpdateBegin(gvec, INSERT_VALUES, SCATTER_FORWARD); assert(ierr == 0);
	ierr = VecGhostUpdateEnd(gvec, INSERT_VALUES, SCATTER_FORWARD); assert(ierr == 0);

	{
		ConstGhostedVecHandler<PetscScalar> gvh(gvec);
		const PetscScalar *const gloc = gvh.getArray();
		for(fint icface = 0; icface < m.gnConnFace(); icface++)
		{
			const auto& tr = truth.at(m.gconnface(icface,4));
			const fint ghostpos = m.gnelem() + icface;
			for(int d = 0; d < NDIM; d++)
				assert(std::fabs(gloc[ghostpos*NVARS+d] - tr[NDIM+d]) < 1e-10);
		}
	}

	ierr = VecDestroy(&gvec); assert(ierr == 0);

	if(mpirank == 0)
		std::cout << "checkGhostConsistency: passed.\n";
}

int main(int argc, char *argv[])
{
	PetscInitialize(&argc, &argv, NULL, NULL);

	if(argc < 2) {
		std::cout << "Not enough arguments!\n";
		PetscFinalize();
		return -1;
	}

	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	const std::string testtype = argv[1];
	std::cout << "Test type is " << testtype << std::endl;

	if(testtype == "checktrivial") {

		if(argc < 2*nranks+3) {
			std::cout << "Not enough arguments!\n";
			PetscFinalize();
			return -1;
		}

		const int basepos = 3;
		const std::string globalmeshfile = argv[basepos];
		std::vector<std::string> localmeshfiles, distfiles;
		for(int i = basepos+1; i < basepos+nranks+1; i++)
			localmeshfiles.push_back(argv[i]);
		for(int i = basepos+nranks+1; i < basepos+2*nranks+1; i++)
			distfiles.push_back(argv[i]);

		assert(localmeshfiles.size() == static_cast<size_t>(nranks));
		assert(distfiles.size() == static_cast<size_t>(nranks));

		checkTrivial(globalmeshfile, localmeshfiles, distfiles);
	}
	else if (testtype == "sanity")
	{
		const std::string algo = argv[2];
		std::cout << " Santiy check for " << algo << " partitioning.." << std::endl;
		const std::string globalmeshfile = argv[3];

		UMesh<freal,NDIM> gm(readMesh(globalmeshfile));
		gm.compute_topological();

		checkConnectedness(gm, algo);
	}
	else if(testtype == "ghostconsistency")
	{
		if(argc < 3) {
			std::cout << "Not enough arguments!\n";
			PetscFinalize();
			return -1;
		}

		const std::string meshfile = argv[2];
		checkGhostConsistency(meshfile);
	}

	PetscFinalize();
	return 0;
}

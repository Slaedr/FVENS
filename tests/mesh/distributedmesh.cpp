/** \file
 * \brief Tests for distributed parallel mesh preprocessing
 */

#undef NDEBUG

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <memory>
#include "utilities/mpiutils.hpp"
#include "mesh/meshpartitioning.hpp"
#include "mesh/ameshutils.hpp"

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

	printf(" Checking neighbours of each cell..\n");
	for(fint iel = 0; iel < lm.gnelem(); iel++)
	{
		bool foundnb = false;
		for(int j = 0; j < lm.gnfael(iel); j++)
		{
			const fint jel = lm.gesuel(iel,j);
			if(jel >= 0 && jel < lm.gnelem())
				foundnb = true;
		}
		assert(foundnb);
	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	if(argc < 2) {
		std::cout << "Not enough arguments!\n";
		MPI_Finalize();
		return -1;
	}

	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	const std::string testtype = argv[1];
	std::cout << "Test type is " << testtype << std::endl;

	if(testtype == "checktrivial") {

		if(argc < 2*nranks+3) {
			std::cout << "Not enough arguments!\n";
			MPI_Finalize();
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
	else if (testtype == "connectedness")
	{
		const std::string algo = argv[2];
		std::cout << " Connectedness check for " << algo << " partitioning.." << std::endl;
		const std::string globalmeshfile = argv[3];

		UMesh<freal,NDIM> gm(readMesh(globalmeshfile));
		gm.compute_topological();

		checkConnectedness(gm, algo);
	}

	MPI_Finalize();
	return 0;
}

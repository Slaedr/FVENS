/** \file
 * \brief Tests for distributed parallel mesh preprocessing
 */

#undef NDEBUG

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "utilities/mpiutils.hpp"
#include "mesh/meshpartitioning.hpp"
#include "mesh/ameshutils.hpp"

using namespace fvens;

/// Gets global element numbering
std::vector<a_int> getElemDist(const a_int nelem, std::ifstream& fin)
{
	std::vector<a_int> elems(nelem);
	std::string dum;
	fin >> dum;
	for(a_int i = 0; i < nelem; i++)
		fin >> elems[i];
	return elems;
}

/// Gets connectivity face data
/// assumes getElemDist has been called using the same ifstream beforehand
amat::Array2d<a_int> getConnMatrix(const a_int nconnface, std::ifstream& fin)
{
	amat::Array2d<a_int> conn;
	if(nconnface > 0)
		conn.resize(nconnface,4);
	std::string dum;
	fin >> dum;
	for(a_int i = 0; i < nconnface; i++)
		for(int j = 0; j < 4; j++)
			fin >> conn(i,j);
	return conn;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	if(argc < 2*nranks+2) {
		std::cout << "Not enough arguments!\n";
		MPI_Finalize();
		return -1;
	}

	const int basepos = 1;

	const std::string globalmeshfile = argv[basepos];
	std::vector<std::string> localmeshfiles, distfiles;
	for(int i = basepos+1; i < basepos+nranks+1; i++)
		localmeshfiles.push_back(argv[i]);
	for(int i = basepos+nranks+1; i < basepos+2*nranks+1; i++)
		distfiles.push_back(argv[i]);

	assert(localmeshfiles.size() == static_cast<size_t>(nranks));
	assert(distfiles.size() == static_cast<size_t>(nranks));

	UMesh2dh<a_real> gm(readMesh(globalmeshfile));
	gm.compute_topological();

	TrivialReplicatedGlobalMeshPartitioner p(gm);
	p.compute_partition();
	const UMesh2dh<a_real> lm = p.restrictMeshToPartitions();

	// Read solution to check against
	const UMesh2dh<a_real> reflm(readMesh(localmeshfiles[rank]));
	std::ifstream fin(distfiles[rank]);
	if(!fin) {
		throw std::runtime_error("File not found!");
	}
	const std::vector<a_int> elemglindices = getElemDist(lm.gnelem(), fin);
	const amat::Array2d<a_int> connface = getConnMatrix(lm.gnConnFace(), fin);
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

			for(a_int i = 0; i < lm.gnelem(); i++) {
				assert(lm.gglobalElemIndex(i) == elemglindices[i]);
			}
			for(a_int i = 0; i < lm.gnConnFace(); i++)
				for(int j = 0; j < 4; j++)
					assert(lm.gconnface(i,j) == connface(i,j));
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

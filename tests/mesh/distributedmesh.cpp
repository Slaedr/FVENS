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

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	const int nranks = get_mpi_size(MPI_COMM_WORLD);

	if(argc < nranks+2) {
		std::cout << "Not enough arguments!\n";
		MPI_Finalize();
		return -1;
	}

	const int basepos = 1;

	const std::string globalmeshfile = argv[basepos];
	std::vector<std::string> localmeshfiles;
	for(int i = basepos+1; i < argc; i++)
		localmeshfiles.push_back(argv[i]);

	assert(localmeshfiles.size() == static_cast<size_t>(nranks));

	UMesh2dh<a_real> gm(readMesh(globalmeshfile));
	gm.compute_topological();

	TrivialReplicatedGlobalMeshPartitioner p(gm);
	p.compute_partition();
	const UMesh2dh<a_real> lm = p.restrictMeshToPartitions();

	UMesh2dh<a_real> reflm(readMesh(localmeshfiles[rank]));

	const std::array<bool,8> isequal = compareMeshes(lm, reflm);

	for(int irnk = 0; irnk < nranks; irnk++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == irnk) {
			assert(isequal[0]);
			assert(isequal[1]);
			assert(isequal[2]);
			assert(isequal[3]);
			assert(isequal[4]);
			assert(isequal[5]);
			assert(isequal[6]);
			assert(isequal[7]);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

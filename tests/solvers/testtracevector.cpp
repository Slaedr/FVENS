/** \file
 * \brief Test for trace vector communication
 */

#undef NDEBUG

#include <iostream>
#include "utilities/mpiutils.hpp"
#include "mesh/ameshutils.hpp"
//#include "mesh/meshpartitioning.hpp"
#include "linalg/tracevector.hpp"

using namespace fvens;

int test(const std::string meshpath)
{
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	// UMesh2dh<a_real> gm(readMesh(meshpath));
	// gm.compute_topological();
	// if(mpirank == 0)
	// 	std::cout << "**" << std::endl;
	// MPI_Barrier(PETSC_COMM_WORLD);

	// // Partition
	// TrivialReplicatedGlobalMeshPartitioner p(gm);
	// p.compute_partition();
	// UMesh2dh<a_real> m = p.restrictMeshToPartitions();
	// int ierr = preprocessMesh<a_real>(m); 
	// fvens_throw(ierr, "Mesh could not be preprocessed!");

	const UMesh2dh<a_real> m = constructMesh(meshpath);

	L2TraceVector<a_real,NVARS> tvec(m);

	// Fill some data

	a_real *const left = tvec.getLocalArrayLeft();
	for(a_int iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		//const a_int icface = iface - m.gConnBFaceStart();
		for(int j = 0; j < NVARS; j++)
			left[iface*NVARS+j] = mpirank*1000/*+m.gconnface(icface,4)*10*/+j;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	tvec.updateSharedFacesBegin();
	std::cout << "Rank " << mpirank << ": ^v" << std::endl;
	tvec.updateSharedFacesEnd();

	// check whether data came from the correct ranks, at least

	a_real *const right = tvec.getLocalArrayRight();

// #ifdef DEBUG
// 	const int mpisize = get_mpi_size(MPI_COMM_WORLD);
// 	for(int irank = 0; irank < mpisize; irank++)
// 	{
// 		MPI_Barrier(MPI_COMM_WORLD);
// 		if(irank == mpirank) {
// 			std::cout << "Rank " << irank << " checking" << std::endl;
// 			for(a_int iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
// 			{
// 				const a_int icface = iface - m.gConnBFaceStart();
// 				const int nbdrank = m.gconnface(icface,2);
// 				for(int j = 0; j < NVARS; j++) {
// 					if(right[iface*NVARS+j] != nbdrank*1000/*+m.gconnface(icface,4)*10*/+j)
// 						std::cout << iface << ": " << right[iface*NVARS+j] << " " << nbdrank*1000+j
// 						          << std::endl;
// 				}
// 			}
// 		}
// 		MPI_Barrier(MPI_COMM_WORLD);
// 	}
// #endif

	MPI_Barrier(MPI_COMM_WORLD);

	for(a_int iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		const a_int icface = iface - m.gConnBFaceStart();
		const int nbdrank = m.gconnface(icface,2);
		for(int j = 0; j < NVARS; j++) {
			assert(right[iface*NVARS+j] == nbdrank*1000/*+m.gconnface(icface,4)*10*/+j);
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 2) {
		std::cout << "A mesh path is required!" << std::endl;
		return -1;
	}

	const std::string mesh = argv[1];

	PetscInitialize(&argc, &argv, NULL, NULL);

	test(mesh);

	PetscFinalize();
	return 0;
}

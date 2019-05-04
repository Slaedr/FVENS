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

	const UMesh<freal,NDIM> m = constructMesh(meshpath);

	L2TraceVector<freal,NVARS> tvec(m);

	// Fill some data

	freal *const left = tvec.getLocalArrayLeft();
	for(fint iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		const fint icface = iface - m.gConnBFaceStart();
		for(int j = 0; j < NVARS; j++)
			left[iface*NVARS+j] = mpirank*1000+m.gconnface(icface,4)*10+j;
	}

	tvec.updateSharedFacesBegin();
	std::cout << "Rank " << mpirank << ": ^v" << std::endl;
	tvec.updateSharedFacesEnd();

	// check whether data came from the correct ranks, at least

	freal *const right = tvec.getLocalArrayRight();

	for(fint iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		const fint icface = iface - m.gConnBFaceStart();
		const int nbdrank = m.gconnface(icface,2);
		for(int j = 0; j < NVARS; j++) {
			assert(right[iface*NVARS+j] == nbdrank*1000+m.gconnface(icface,4)*10+j);
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

/** \file
 * \brief Test for trace vector communication
 */

#undef NDEBUG

#include <iostream>
#include "utilities/mpiutils.hpp"
#include "mesh/ameshutils.hpp"
#include "linalg/tracevector.hpp"

using namespace fvens;

int test(const std::string meshpath)
{
	const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	const UMesh2dh<a_real> m = constructMesh(meshpath);
	L2TraceVector<a_real,NVARS> tvec(m);

	// Fill some data

	a_real *const left = tvec.getLocalArrayLeft();
	for(a_int iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		for(int j = 0; j < NVARS; j++)
			left[iface*NVARS+j] = mpirank+1;
	}

	tvec.updateSharedFacesBegin();
	tvec.updateSharedFacesEnd();

	// check whether data came from the correct ranks, at least

	a_real *const right = tvec.getLocalArrayRight();
	for(a_int iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		const a_int icface = iface - m.gConnBFaceStart();
		const int nbdrank = m.gconnface(icface,2);
		for(int j = 0; j < NVARS; j++)
			assert(right[iface*NVARS+j] == nbdrank+1);
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

#undef NDEBUG
#include <iostream>
#include "mesh/ameshutils.hpp"
#include "testgradientschemes.hpp"

using namespace fvens;
using namespace fvens_tests;

int main(int argc, char *argv[])
{
	if(argc < 4) {
		std::cout << "Need mesh file, gradient scheme and test type.";
		std::exit(-1);
	}

	PetscInitialize(&argc,&argv,NULL,NULL);

	int ierr = 0;

	UMesh2dh<a_real> m(readMesh(argv[1]));
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();

	{
		TestSpatial ts(&m);

		const std::string testtype = argv[3];
		if(testtype == "1exact")
			ierr = ts.test_oneExact(argv[2]);
		else {
			std::cout << "Invalid test!\n";
			std::exit(-2);
		}
	}

	PetscFinalize();
	return ierr;
}

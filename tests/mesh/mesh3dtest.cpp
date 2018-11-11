#undef NDEBUG

#include "mesh/amesh.hpp"

using namespace fvens;

int main(int argc, char *argv[])
{
	UMesh<double,3> m;
	m.readMesh("../common-input/box-tet.msh");

	assert(m.gnelem() == 364-168);
	assert(m.gnpoin() == 87);
	assert(m.gnface() == 168);
	assert(m.gndtag() == 2);
	assert(m.gnbtag() == 2);

	assert(m.gbface(1,m.gnnodeBFace(1))==2);

	return 0;
}

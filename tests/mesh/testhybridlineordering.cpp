#undef NDEBUG

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include "mesh/meshordering.hpp"
#include "mesh/details_lineordering.hpp"
#include "mesh/ameshutils.hpp"

using namespace fvens;

int main(int argc, char *argv[])
{
	char help[] = "Need \n- a mesh file,\n- a file containing lines - one mesh line on each file line,\
\n- the local anisotropy threshold, \n- the PETSc reordering, in that order.\n\
The solution file must have cell-numbers exactly according to the msh file.\n";
	if(argc < 3) {
		printf("%s",help);
		exit(-1);
	}

	PetscInitialize(&argc, &argv, NULL, help);

	const std::string meshfile = argv[1];
	const std::string solnfile = argv[2];
	const std::string petscordering = argv[3];
	const double threshold = std::stod(argv[4]);

	UMesh2dh<a_real> m = constructMesh(meshfile);
	const a_int orignelem = m.gnelem();
	int ierr = preprocessMesh<a_real>(m);

	const std::vector<a_int> lorder = getHybridLineOrdering<a_real>(m, threshold, petscordering.c_str());
	assert(lorder.size() == static_cast<size_t>(orignelem));

	printf("Computed order is\n");
	for(a_int i = 0; i < m.gnelem(); i++) {
		printf(" %d ", lorder[i]+1+m.gnbface());
	}
	printf("\n"); fflush(stdout);

	std::vector<a_int> solnord(m.gnelem());
	std::ifstream infile(solnfile);
	for(a_int i = 0; i < m.gnelem(); i++)
		infile >> solnord[i];

	for(a_int i = 0; i < m.gnelem(); i++)
		assert(lorder[i]+1+m.gnbface() == solnord[i]);

	PetscFinalize();
	return ierr;
}

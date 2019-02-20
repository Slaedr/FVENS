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
	char help[] = "Need \n- a mesh file,\n- a file containing lines - one mesh line on each file line\
and \n- the local anisotropy threshold, in that order.\n\
The solution file must have cell-numbers exactly according to the msh file.\n";
	if(argc < 3) {
		printf("%s",help);
		exit(-1);
	}

	PetscInitialize(&argc, &argv, NULL, help);

	const std::string meshfile = argv[1];
	const std::string solnfile = argv[2];
	const double threshold = std::stod(argv[3]);

	UMesh2dh<a_real> m = constructMesh(meshfile);
	int ierr = preprocessMesh<a_real>(m);

	const LineConfig lc = findLines(m, threshold);
	std::cout << "Found " << lc.lines.size() << " lines." << std::endl;

	std::vector<std::vector<int>> solines;

	std::ifstream infile(solnfile);
	std::string fileline;
	while(std::getline(infile, fileline)) {
		std::stringstream ss(fileline);
		int a;
		std::vector<int> meshline;
		while(ss >> a)
			meshline.push_back(a);
		solines.push_back(meshline);
	}
	infile.close();

	std::cout << "Reference solution is \n";
	for(size_t i = 0; i < solines.size(); i++)
	{
		for(size_t j = 0; j < solines[i].size(); j++)
			std::cout << " " << solines[i][j];
		std::cout << std::endl;
	}
	std::cout << std::endl;

	assert(lc.lines.size() == solines.size());
	for(size_t i = 0; i < solines.size(); i++)
	{
		assert(lc.lines[i].size() == solines[i].size());
		for(size_t j = 0; j < solines[i].size(); j++)
			assert(lc.lines[i][j]+m.gnbface()+1 == solines[i][j]);
	}

	PetscFinalize();
	return ierr;
}

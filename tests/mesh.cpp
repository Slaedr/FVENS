#include "../amesh2dh.hpp"

using namespace acfd;

int test_periodic_map(const std::string mfile, const int bcm, const int axis)
{
	UMesh2dh m;
	m.readMesh(mfile);
	m.compute_topological();
	m.compute_face_data();
	m.compute_periodic_map(bcm,axis);

	// map intfac faces to mesh faces for testing
	m.compute_boundary_maps();

	const int numfaces = 5;
	a_int faces1[] = {8,9,10,11,12};
	a_int faces2[] = {25,24,23,22,21};

	int ierr = 0;

	for(int i = 0; i < numfaces; i++) {
		if(m.gperiodicmap(m.gifbmap(faces1[i])) != m.gifbmap(faces2[i])) {
			ierr = 1;
			std::cerr << "  Face " << faces1[i] << " failed!\n";
		}
		if(m.gperiodicmap(m.gifbmap(faces2[i])) != m.gifbmap(faces1[i])) {
			ierr = 1;
			std::cerr << "  Face " << faces1[i] << " failed!\n";
		}
	}

	return ierr;
}

int main(int argc, char *argv[])
{
	if(argc < 2) {
		std::cout << "Not enough command-line arguments!\n";
		return -2;
	}

	int finerr = 0;

	int err = test_periodic_map(argv[1], 4, 0);
	if(err) std::cerr << " Periodic map test failed!\n";
	finerr = finerr || err;

	return finerr;
}

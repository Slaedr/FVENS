#undef NDEBUG
#include "../test.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "mesh/mesh.hpp"
#include "mesh/ameshutils.hpp"

#include <cassert>

using namespace fvens;

/// For each element in the esup list of a point, check if that point is actually part of each
///  of those elements.
template<typename scalar>
int test_topology_internalconsistency_esup(const UMesh<scalar,NDIM>& m)
{
	for(fint ipoin = 0; ipoin < m.gnpoin(); ipoin++)
	{
		for(int ielind = m.gesup_p(ipoin); ielind < m.gesup_p(ipoin+1); ielind++)
		{
			const fint iel = m.gesup(ielind);
			if(iel >= m.gnelem())
				continue;

			bool found = false;
			for(int jp = 0; jp < m.gnnode(iel); jp++)
				if(m.ginpoel(iel,jp) == ipoin)
					found = true;

			TASSERT(found);
		}
	}
	return 0;
}

/// For each mesh face, check if the left and right elements actually contain that face
template<typename scalar>
int test_topology_internalconsistency_intfac(const UMesh<scalar,NDIM>& m)
{
	static_assert(NDIM==2, "Only 2D meshes are currently supported!");

	for(fint iface = m.gPhyBFaceStart(); iface < m.gPhyBFaceEnd(); iface++)
	{
		const fint lelem = m.gintfac(iface,0);
		const EIndex locface = m.getFaceEIndex(false, iface, lelem);
		assert(locface >= 0);
	}
	for(fint iface = m.gSubDomFaceStart(); iface < m.gSubDomFaceEnd(); iface++)
	{
		const fint lelem = m.gintfac(iface,0);
		const fint relem = m.gintfac(iface,1);
		const EIndex locface1 = m.getFaceEIndex(false, iface, lelem);
		assert(locface1 >= 0);
		const EIndex locface2 = m.getFaceEIndex(false, iface, relem);
		assert(locface2 >= 0);
	}
	for(fint iface = m.gConnBFaceStart(); iface < m.gConnBFaceEnd(); iface++)
	{
		const fint lelem = m.gintfac(iface,0);
		const EIndex locface = m.getFaceEIndex(false, iface, lelem);
		assert(locface >= 0);
	}
	return 0;
}

template<typename scalar>
int test_periodic_map(UMesh<scalar,NDIM>& m, const int bcm, const int axis)
{
	m.compute_face_data();
	m.compute_periodic_map(bcm,axis);

	const int numfaces = 5;
	const fint faces1[] = {8,9,10,11,12};
	const fint faces2[] = {25,24,23,22,21};

	int ierr = 0;

	for(int i = 0; i < numfaces; i++) {
		assert(m.gintfac(m.gPhyBFaceStart()+faces1[i],1) == m.gintfac(m.gPhyBFaceStart()+faces2[i],0));
		assert(m.gintfac(m.gPhyBFaceStart()+faces1[i],0) == m.gintfac(m.gPhyBFaceStart()+faces2[i],1));
	}

	return ierr;
}

template<typename scalar>
int test_levelscheduling(const UMesh<scalar,NDIM>& m, const std::string levelsfile)
{
	std::vector<fint> levels = levelSchedule(m);
	const int nlevels = (int)(levels.size()-1);

	std::ifstream lfile;
	try {
		lfile.open(levelsfile);
	} catch (std::exception& e) {
		throw e;
	}

	int real_nlevels;
	lfile >> real_nlevels;
	std::vector<fint> real_levels(real_nlevels+1);
	for(int i = 0; i < real_nlevels+1; i++)
		lfile >> real_levels[i];

	lfile.close();

	for(int i = 0; i < real_nlevels+1; i++)
		std::cout << real_levels[i] << "  " <<  levels[i] << std::endl;

	TASSERT(real_nlevels == nlevels);
	std::cout << "tested num levels\n";

	for(int i = 0; i < real_nlevels+1; i++)
		TASSERT(real_levels[i] == levels[i]);

	return 0;
}

template<typename scalar>
int test_levelscheduling_internalconsistency(const UMesh<scalar,NDIM>& m)
{
	std::vector<fint> levels = levelSchedule(m);
	const int nlevels = (int)(levels.size()-1);

	for(int ilevel = 0; ilevel < nlevels; ilevel++)
	{
		std::vector<int> marked(m.gnelem(), 0);

		// mark all neighbors of all cells in this level
		// but not the cells in the level themselves
		for(int icell = levels[ilevel]; icell < levels[ilevel+1]; icell++)
		{
			for(int iface = 0; iface < m.gnfael(icell); iface++) {
				if(m.gesuel(icell,iface) < m.gnelem())
					marked[m.gesuel(icell,iface)] = 1;
			}
		}

		// check if the cells in this levels have not been marked
		for(int icell = levels[ilevel]; icell < levels[ilevel+1]; icell++) {
			if(marked[icell]) std::cout << "Problem with " << icell << std::endl;
			TASSERT(!marked[icell]);
		}
	}
	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 3) {
		std::cout << "Not enough command-line arguments!\n";
		return -2;
	}

	MPI_Init(&argc, &argv);

	std::string whichtest = argv[1];
	int err = 0;

	const MeshData md = readMesh(argv[2]);
	UMesh<freal,NDIM> m(md);

	m.compute_topological();

	if(whichtest == "esup") {
		err = test_topology_internalconsistency_esup(m);
	}
	else if(whichtest == "intfac") {
		err = test_topology_internalconsistency_intfac(m);
	}
	else if(whichtest == "periodic") {
		err = test_periodic_map(m, 4, 0);
		if(err) std::cerr << " Periodic map test failed!\n";
	}
	else if(whichtest == "levelschedule") {
		if(argc < 4) {
			std::cout << "Not enough command-line arguments!\n";
			return -2;
		}
		err = test_levelscheduling(m, argv[3]);
	}
	else if(whichtest == "levelscheduleInternal") {
		err = test_levelscheduling_internalconsistency(m);
	}
	else
		throw "Invalid test";

	MPI_Finalize();
	return err;
}

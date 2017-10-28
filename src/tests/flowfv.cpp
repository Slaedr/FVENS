#include <string>
#include "../aspatial.hpp"

using namespace acfd;

class TestFlowFVGeneral : public FlowFV<true,false>
{
public:
	TestFlowFVGeneral(const UMesh2dh *const mesh, const FlowPhysicsConfig& pconf,
			const FlowNumericsConfig& nconf)
	: FlowFV<true,false>(mesh, pconf, nconf)
	{ }
	
	/// Tests whether the LLF flux is zero at solid walls
	/** \param u Interior state for boundary cells
	 */
	int testWalls(const a_real u[NVARS]) const
	{
		int ierr = 0;

		for(int iface = 0; iface < m->gnbface(); iface++)
		{
			a_real ug[NVARS];
			a_real n[NDIM];
			for(int i = 0; i < NDIM; i++)
				n[i] = m->ggallfa(iface,i);

			compute_boundary_state(iface, n, u, ug);

			a_real flux[NVARS];
			inviflux->get_flux(u,ug,n,flux);
			
			if(m->gintfacbtags(iface,0) == pconfig.adiabaticwall_id)
			{
				if(std::fabs(flux[0]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal mass flux at adiabatic wall is nonzero!\n";
				}

				if(std::fabs(flux[NVARS-1]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal energy flux at adiabatic wall is nonzero!\n";
				}
			}

			if(m->gintfacbtags(iface,0) == pconfig.isothermalwall_id)
				if(std::fabs(flux[0]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal mass flux at isothermal wall is nonzero!\n";
				}

			if(m->gintfacbtags(iface,0) == pconfig.isothermalbaricwall_id)
				if(std::fabs(flux[0]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal mass flux at isothermalbaric wall is nonzero!\n";
				}
		}

		return ierr;
	}

protected:
	using FlowFV<true,false>::compute_boundary_state;
	using FlowFV<true,false>::inviflux;
};

int main()
{
	if(argc < 2) {
		std::cout << "Not enough command-line arguments!\n";
		return -2;
	}

	int finerr = 0;

	std::string meshfile = argv[1];
	UMesh2dh m;
	m.readMesh(meshfile);
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();
}

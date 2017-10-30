#include <string>
#include "../aspatial.hpp"
#include "../autilities.hpp"

using namespace acfd;

class TestFlowFVGeneral : public FlowFV<true,false>
{
public:
	TestFlowFVGeneral(const UMesh2dh *const mesh, const FlowPhysicsConfig& pconf,
			const FlowNumericsConfig& nconf)
	: FlowFV<true,false>(mesh, pconf, nconf)
	{ }
	
	/// Tests whether the inviscid mass flux is zero at solid walls
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

			compute_boundary_state(iface, u, ug);

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
			
			if(m->gintfacbtags(iface,0) == pconfig.slipwall_id)
			{
				if(std::fabs(flux[0]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal mass flux at slip wall is nonzero!\n";
				}

				if(std::fabs(flux[NVARS-1]) > ZERO_TOL) {
					ierr = 1;
					std::cerr << "! Normal energy flux at slip wall is nonzero!\n";
				}
			}
		}

		return ierr;
	}

protected:
	using FlowFV<true,false>::compute_boundary_state;
	using FlowFV<true,false>::inviflux;
};

/** The first command line argument is the control file.
 * The second is a string that decides which test to perform.
 * Currently avaiable:
 * - 'wall_boundaries': Tests whether certain components of the numerical inviscid flux
 *     are zero for the 3 types of solid walls - adiabatic, isothermal and slip.
 */
int main(const int argc, const char *const argv[])
{
	if(argc < 3) {
		std::cout << "Not enough command-line arguments!\n";
		return -2;
	}

	int finerr = 0;
	
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv);
	std::string testchoice = argv[2];

	UMesh2dh m;
	m.readMesh(opts.meshfile);
	m.compute_topological();
	m.compute_areas();
	m.compute_face_data();

	const FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	const FlowNumericsConfig nconf = extract_spatial_numerics_config(opts);
	TestFlowFVGeneral testfv(&m, pconf, nconf);

	if(testchoice == "wall_boundaries") {
		const a_real p_nondim = 10.0;
		const a_real u[NVARS] = {1.0, 0.5, 0.5, p_nondim/(opts.gamma-1.0) + 0.5*0.5 };
		
		int err = testfv.testWalls(u);
		finerr = finerr || err;
	}

	return finerr;
}

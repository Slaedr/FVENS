#include <string>
#include "../autilities.hpp"
#include "testflowspatial.hpp"

using namespace acfd;

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
		
	FlowPhysicsConfig pconf = extract_spatial_physics_config(opts);
	FlowNumericsConfig nconf = extract_spatial_numerics_config(opts);

	if(testchoice == "wall_boundaries") 
	{
		TestFlowFV testfv(&m, pconf, nconf);
		const std::array<a_real,NVARS> u = get_test_state();
		
		/* tests whether mass flux is zero at solid walls and whether
		 * energy flux is zero at adiabatic and slip walls.
		 */
		int err = testfv.testWalls(&u[0]);
		finerr = finerr || err;
	}

	if(testchoice == "numerical_flux")
	{
		if(argc < 4) {
			std::cerr << "Not enough command-line arguments!\n";
			return -2;
		}
		std::string testflux = argv[3];
		nconf.conv_numflux = testflux;
		nconf.conv_numflux_jac = testflux;
		TestFlowFV testfv(&m, pconf, nconf);
		
		const std::array<a_real,NVARS> u = get_test_state();
	
		/* tests whether mass flux is zero at solid walls and whether
		 * energy flux is zero at adiabatic and slip walls.
		 */
		int err = testfv.testWalls(&u[0]);
		finerr = finerr || err;
	}

	return finerr;
}

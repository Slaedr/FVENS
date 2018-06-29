/** \file testwallbcs.hpp
 * \brief Specifies a class for testing wall boundary condition of a flow problem
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef FVENS_TEST_WALLBCS_H
#define FVENS_TEST_WALLBCS_H

#include <string>
#include <array>
#include "utilities/aoptionparser.hpp"
#include "spatial/aspatial.hpp"

namespace fvens {
namespace fvens_tests {

class TestFlowFV : public FlowFV<true,false>
{
public:
	TestFlowFV(const UMesh2dh<a_real> *const mesh, const FlowPhysicsConfig& pconf,
			const FlowNumericsConfig& nconf)
	: FlowFV<true,false>(mesh, pconf, nconf)
	{ }
	
	/// Tests whether the inviscid numerical mass flux is zero at solid walls
	/** 'Solid walls' include slip walls, adiabatic walls, and isothermal walls.
	 * \param u Interior state for boundary cells
	 */
	int testWalls(const a_real *const u) const;

protected:
	using FlowFV<true,false>::compute_boundary_state;
	using FlowFV<true,false>::inviflux;
};

/// Returns a state vector in conserved variables that can be used in testing
std::array<a_real,NVARS> get_test_state();

}
}
#endif

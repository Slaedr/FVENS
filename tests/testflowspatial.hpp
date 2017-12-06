/** \file testflowspatial.hpp
 * \brief Specifies a class for testing various aspects of the spatial discretization of a flow problem
 * \author Aditya Kashi
 * \date 2017-10
 */

#ifndef TESTFLOWSPATIAL_H
#define TESTFLOWSPATIAL_H

#include <string>
#include <array>
#include "../autilities.hpp"
#include "../aspatial.hpp"

namespace acfd {

class TestFlowFV : public FlowFV<true,false>
{
public:
	TestFlowFV(const UMesh2dh *const mesh, const FlowPhysicsConfig& pconf,
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
#endif

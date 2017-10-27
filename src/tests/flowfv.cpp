#include "../aspatial.hpp"

using namespace acfd;

class TestFlowFVGeneral : public FlowFV<true,false>
{
public:
	TestFlowFVGeneral()
	: FlowFV<true,false>()
	{ }

	
	/// Tests whether the LLF flux is zero at solid walls
	void testWalls(const a_real uin[NVARS])
	{
		// TODO
	}
};

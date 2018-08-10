/** \file abctypemap.cpp
 * \brief Initialization for the BC type string map
 * \author Aditya Kashi
 */

#include <iostream>
#include "abctypemap.hpp"

namespace fvens {

/// Factory for the BC type map
static boost::bimap<BCType, std::string> createBCTypeMap();

const boost::bimap<BCType, std::string> bcTypeMap = createBCTypeMap();

boost::bimap<BCType, std::string> createBCTypeMap()
{
	typedef boost::bimap<BCType, std::string>::value_type rel_entry;
	boost::bimap<BCType, std::string> bmp;

	bmp.insert(rel_entry(SLIP_WALL_BC, "slipwall"));
	bmp.insert(rel_entry(ISOTHERMAL_WALL_BC, "isothermalwall"));
	bmp.insert(rel_entry(ADIABATIC_WALL_BC, "adiabaticwall"));
	bmp.insert(rel_entry(FARFIELD_BC, "farfield"));
	bmp.insert(rel_entry(INFLOW_OUTFLOW_BC, "inflowoutflow"));
	bmp.insert(rel_entry(SUBSONIC_INFLOW_BC, "subsonicinflow"));
	bmp.insert(rel_entry(EXTRAPOLATION_BC, "extrapolation"));
	bmp.insert(rel_entry(PERIODIC_BC, "periodic"));

	return bmp;
}

}

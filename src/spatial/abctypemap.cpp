/** \file abctypemap.cpp
 * \brief Initialization function definition for the BC type string map
 * \author Aditya Kashi
 */

#include "abctypemap.hpp"

namespace fvens {

boost::bimap<BCType, std::string> bcTypeMap;

void setBCTypeMap()
{
	typedef boost::bimap<BCType, std::string>::value_type rel_entry;
	bcTypeMap.insert(rel_entry(SLIP_WALL_BC, "slipwall"));
	bcTypeMap.insert(rel_entry(ISOTHERMAL_WALL_BC, "isothermalwall"));
	bcTypeMap.insert(rel_entry(ADIABATIC_WALL_BC, "adiabaticwall"));
	bcTypeMap.insert(rel_entry(FARFIELD_BC, "farfield"));
	bcTypeMap.insert(rel_entry(INFLOW_OUTFLOW_BC, "inflowoutflow"));
	bcTypeMap.insert(rel_entry(SUBSONIC_INFLOW_BC, "subsonicinflow"));
	bcTypeMap.insert(rel_entry(EXTRAPOLATION_BC, "extrapolation"));
	bcTypeMap.insert(rel_entry(PERIODIC_BC, "periodic"));
}

}

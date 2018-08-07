/** \file abctypes.hpp
 * \brief Enumeration of types of boundary conditions
 * \author Aditya Kashi
 */

#ifndef FVENS_BCTYPES_H
#define FVENS_BCTYPES_H

namespace fvens {

/// The types of boundary condition a boundary face can have
enum BCType {
	SLIP_WALL_BC,
	FARFIELD_BC,
	INFLOW_OUTFLOW_BC,
	SUBSONIC_INFLOW_BC,
	EXTRAPOLATION_BC,
	PERIODIC_BC,
	ISOTHERMAL_WALL_BC,
	ADIABATIC_WALL_BC
};

}
#endif

/** \file abc.hpp
 * \brief Boundary conditions management
 * \author Aditya Kashi
 * \date 2018-05
 * \todo Implement!
 */

#ifndef FVENS_BC_H
#define FVENS_BC_H

#include "aphysics.hpp"

namespace acfd {

/// The types of boundary condition a boundary face can have
enum BCType {
	SLIP_WALL_BC,
	FARFIELD_BC,
	INFLOW_OUTFLOW_BC,
	EXTRAPOLATION_BC,
	PERIODIC_BC,
	ISOTHERMAL_WALL_BC,
	ADIABATIC_WALL_BC
};

/// Abstract class for storing the details and providing functionality for one type if BC
/** Each FlowBC (derived class) object is associated with an integer [tag](\ref UMesh2dh::bface)
 * which, in turn, is associated with a collection of boundary faces on which the BC is to be applied.
 * Note that multiple instantiations of a single BC type may be required for different boundary
 * values.
 */
template <typename scalar>
class FlowBC
{
public:
	FlowBC(const int face_id, const IdealGasPhysics& gasphysics);

	/// Computes the ghost state given the interior state and normal vector
	virtual void computeGhostState(const scalar *const uin, const scalar *const n,
	                               scalar *const ughost) const = 0;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	virtual void computeJacobian(const scalar *const uin, const scalar *const n,
	                             scalar *const dugdui) const = 0;

protected:
	/// Tag index of mesh faces on which this BC is to be applied
	int faceid;

	/// Thermodynamic and some mechanical properties of the fluid
	const IdealGasPhysics& phy;
};

template <typename scalar>
class SlipWall : public FlowBC
{
public:
	SlipWall(const int face_id, const IdealGasPhysics& gasphysics);

	/// Computes the ghost state given the interior state and normal vector
	virtual void computeGhostState(const scalar *const uin, const scalar *const n,
	                               scalar *const ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	virtual void computeJacobian(const scalar *const uin, const scalar *const n,
	                             scalar *const dugdui) const;
};

/// Currently, this is a pressure-imposed outflow and all-imposed inflow BC
template <typename scalar>
class InOutFlow : public FlowBC
{
public:
	SlipWall(const int face_id, const IdealGasPhysics& gasphysics, const scalar ufar[NVARS]);

	/// Computes the ghost state given the interior state and normal vector
	virtual void computeGhostState(const scalar *const uin, const scalar *const n,
	                               scalar *const ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	virtual void computeJacobian(const scalar *const uin, const scalar *const n,
	                             scalar *const dugdui) const;
};

}
#endif

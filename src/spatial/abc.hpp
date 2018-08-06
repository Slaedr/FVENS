/** \file abc.hpp
 * \brief Boundary conditions management
 * \author Aditya Kashi
 * \date 2018-05
 */

#ifndef FVENS_BC_H
#define FVENS_BC_H

#include <map>
#include "physics/aphysics.hpp"

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

/// Converts a string (lower case) to the corresponding BCType
BCType getBCTypeFromString(const std::string bct);

/// Collection of options describing boundary conditions
struct FlowBCConfig {
	int isothermalwall_id;            ///< Boundary marker for isothermal no-slip wall
	int adiabaticwall_id;             ///< Marker for adiabatic no-slip wall
	int isothermalbaricwall_id;       ///< Marker for isothermal fixed-pressure no-slip wall
	int slipwall_id;                  ///< Marker for slip wall
	int farfield_id;                  ///< Marker for far-field boundary
	int inflowoutflow_id;             ///< Marker for inflow/outflow boundary
	int subsonicinflow_id;            ///< Marker for subsonic inflow
	int extrapolation_id;             ///< Marker for an extrapolation boundary
	int periodic_id;                  ///< Marker for periodic boundary
	a_real subsonicinflow_ptot;       ///< Total non-dimensional pressure at subsonic inflow
	a_real subsonicinflow_ttot;       ///< Total temperature at subsonic inflow
	a_real isothermalwall_temp;       ///< Temperature at isothermal wall
	a_real isothermalwall_vel;        ///< Tangential velocity at isothermal wall 
	a_real adiabaticwall_vel;         ///< Tangential velocity at adiabatic wall
	a_real isothermalbaricwall_temp;  ///< Temperature at isothermal fixed-pressure wall
	a_real isothermalbaricwall_vel;   ///< Tangential velocity at isothermal pressure wall
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
	/// Set up a flow boundary condition
	/** \param bc_tag The boundary marker tag for which we want this BC to apply
	 * \param gasphysics Context describing some properties of the gas
	 */
	FlowBC(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics);

	virtual ~FlowBC();

	/// Return the boundary marker tag that this BC context applies to
	int bctag() const { return btag; }

	/// Computes the ghost state given the interior state and normal vector
	/** \param uin Interior conserved state
	 * \param n Unit normal vector
	 * \param ughost Ghost (conserved) state (on output)
	 */
	virtual void computeGhostState(const scalar *const uin, const scalar *const n,
	                               scalar *const __restrict ughost) const = 0;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	/** \param uin Interior conserved state
	 * \param n Unit normal vector
	 * \param [in,out] ug Ghost conserved state
	 * \param [in,out] dugdui Jacobian of ghost state w.r.t. interior state (on output)
	 */
	virtual void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                          scalar *const __restrict ug,
	                                          scalar *const __restrict dugdui) const = 0;

protected:
	/// Tag index of mesh faces on which this BC is to be applied
	int btag;

	/// Thermodynamic and some mechanical properties of the fluid
	const IdealGasPhysics<scalar>& phy;
};

/// Currently, this is a pressure-imposed outflow and all-imposed inflow BC
/** This "inflow-outflow" BC assumes we know the state at the inlet is 
 * the free-stream state with certainty,
 * while the state at the outlet is not certain to be the free-stream state. 
 * If so, we can just impose free-stream conditions for the ghost cells of inflow faces.
 *
 * The outflow boundary condition corresponds to Sec 2.4 "Pressure outflow boundary condition"
 * in the paper \cite carlson_bcs. It assumes that the flow at the outflow boundary is
 * isentropic.
 *
 * Whether the flow is subsonic or supersonic at the boundary
 * is decided by interior value of the Mach number.
 */
template <typename scalar>
class InOutFlow : public FlowBC<scalar>
{
public:
	/// Setup inflow-outflow BC 
	/** \sa FlowBC::FlowBC
	 * \param ufar Far-field state
	 */
	InOutFlow(const int face_id, const IdealGasPhysics<scalar>& gasphysics,
	          const std::array<scalar,NVARS>& u_far);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	const std::array<scalar,NVARS> uinf;
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
};

/// Normal inflow BC with total pressure and total temperature specified
/** There are two sources: \cite carlson_bcs section 2.7, and \cite blazek, section 8.4.
 * This is mainly based on the latter (Blazek). One difference though is that here,
 * the flow is constrained normal to the boundary.
 */
template <typename scalar>
class InFlow : public FlowBC<scalar>
{
public:
	/// Setup inflow BC 
	/** \sa FlowBC::FlowBC
	 * \param totalpressure Non-dimensional total pressure at inflow
	 * \param totaltemperature Total temperature at inflow
	 */
	InFlow(const int face_id, const IdealGasPhysics<scalar>& gasphysics,
	       const scalar totalpressure, const scalar totaltemperature); 

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	const scalar ptotal;
	const scalar ttotal;
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
};

/// Simply sets the ghost state as the given free-stream state
template <typename scalar>
class Farfield : public FlowBC<scalar>
{
public:
	/// Setup farfield BC 
	/** \sa FlowBC::FlowBC
	 * \param ufar Far-field state
	 */
	Farfield(const int face_id, const IdealGasPhysics<scalar>& gasphysics,
	         const std::array<scalar,NVARS>& u_far);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	const std::array<scalar,NVARS> uinf;
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
};

/// Simply sets the ghost state as the interior state
template <typename scalar>
class Extrapolation : public FlowBC<scalar>
{
public:
	/// Setup extrapolation BC
	/** \sa FlowBC::FlowBC
	 */
	Extrapolation(const int face_id, const IdealGasPhysics<scalar>& gasphysics);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
};

/// Slip wall BC for Euler equations
template <typename scalar>
class Slipwall : public FlowBC<scalar>
{
public:
	/// Setup slip wall BC 
	/** \sa FlowBC::FlowBC
	 */
	Slipwall(const int face_id, const IdealGasPhysics<scalar>& gasphysics);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
};

/// No-slip adiabatic wall BC for 2D NS equations
template <typename scalar>
class Adiabaticwall2D : public FlowBC<scalar>
{
	static_assert(NDIM == 2, "Adiabaticwall2D is only defined for 2D cases.");
public:
	/// Setup adiabatic no-slip wall BC 
	/** \sa FlowBC::FlowBC
	 */
	Adiabaticwall2D(const int face_id, const IdealGasPhysics<scalar>& gasphysics,
	                const a_real wall_tangential_velocity);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
	const a_real tangvel;
};

/// No-slip isothermal wall BC for 2D NS equations
template <typename scalar>
class Isothermalwall2D : public FlowBC<scalar>
{
	static_assert(NDIM == 2, "Isothermalwall2D is only defined for 2D cases.");
public:
	/// Setup isothermal no-slip wall BC 
	/** \sa FlowBC::FlowBC
	 */
	Isothermalwall2D(const int face_id, const IdealGasPhysics<scalar>& gasphysics,
	                 const a_real wall_tangential_velocity, const a_real wall_temperature);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const scalar *const uin, const scalar *const n,
	                                  scalar *const __restrict ug,
	                                  scalar *const __restrict dugdui) const;

protected:
	using FlowBC<scalar>::btag;
	using FlowBC<scalar>::phy;
	const a_real tangvel;
	const a_real walltemperature;
};

/// Create a set of pointers to immutable boundary condition objects, possibly of different types
/** The BC objects are mapped to their corresponding boundary markers specified in the control file.
 * \param conf Boundary condition parameters read from control file
 * \param physics Gas properties
 * \param uinf Free-stream state in conserved variables
 */
template <typename scalar>
std::map<int,const FlowBC<scalar>*> create_const_flowBCs(const FlowBCConfig& conf,
                                                         const IdealGasPhysics<scalar>& physics,
                                                         const std::array<a_real,NVARS>& uinf);

}
#endif

/** \file abc.hpp
 * \brief Boundary conditions management
 * \todo TODO: Do something about periodic boundary conditions
 * \author Aditya Kashi
 * \date 2018-05
 *
 * This file is part of FVENS.
 *   FVENS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   FVENS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with FVENS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FVENS_BC_H
#define FVENS_BC_H

#include <map>
#include "abctypes.hpp"
#include "physics/aphysics.hpp"

namespace fvens {

/// Definition of boundary condition at one particular boundary
/** This is essentially raw data read from the control file.
 */
struct FlowBCConfig
{
	int bc_tag;                   ///< Boundary marker in mesh file
	BCType bc_type;               ///< Type of boundary
	std::vector<freal> bc_vals;  ///< Boundary value(s)
	std::vector<int> bc_opts;     ///< Other info needed by boundary condition
};

/// Abstract class for storing the details and providing functionality for one type if BC
/** Each FlowBC (derived class) object is associated with an integer [tag](\ref UMesh2dh::bface)
 * which, in turn, is associated with a collection of boundary faces on which the BC is to be applied.
 * Note that multiple objects of a single BC type may be required for different boundary
 * values.
 *
 * Template parameter scalar is the type used for computing the boundary value only. Type j_real is
 * is used for computing Jacobian of the boundary state w.r.t. the interior state. By default,
 * j_real is set the same as \ref freal, the default type for the residual computation.
 */
template <typename scalar, typename j_real = freal>
class FlowBC
{
public:
	/// Set up a flow boundary condition
	/** \param bc_tag The boundary marker tag for which we want this BC to apply
	 * \param gasphysics Context describing some properties of the gas
	 */
	FlowBC(const BCType btype, const int bc_tag, const IdealGasPhysics<scalar>& gasphysics);

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
	virtual void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                          j_real *const __restrict ug,
	                                          j_real *const __restrict dugdui) const = 0;

	/// Type of boundary condition
	const BCType bctype;

protected:
	/// Tag index of mesh faces on which this BC is to be applied
	const int btag;

	/// Thermodynamic and some mechanical properties of the fluid
	const IdealGasPhysics<scalar>& phy;
	/// Thermodynamic and some mechanical properties of the fluid - used for Jacobian computation
	///  This is set to be a copy of phy, but with a different scalar type from scalar.
	const IdealGasPhysics<j_real> jphy;
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
template <typename scalar, typename j_real = freal>
class InOutFlow : public FlowBC<scalar,j_real>
{
public:
	/// Setup inflow-outflow BC
	/** \sa FlowBC::FlowBC
	 * \param ufar Far-field state
	 */
	InOutFlow(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	          const std::array<freal,NVARS>& u_far);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	const std::array<freal,NVARS> uinf;
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
};

/// Normal subsonic inflow BC with total pressure and total temperature specified
/** There are two sources: \cite carlson_bcs section 2.7, and \cite blazek, section 8.4.
 * This is mainly based on the latter (Blazek). One difference though is that here,
 * the flow is constrained normal to the boundary.
 */
template <typename scalar, typename j_real = freal>
class InFlow : public FlowBC<scalar,j_real>
{
public:
	/// Setup inflow BC
	/** \sa FlowBC::FlowBC
	 * \param totalpressure Non-dimensional total pressure at inflow
	 * \param totaltemperature Total temperature at inflow
	 */
	InFlow(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	       const freal totalpressure, const freal totaltemperature);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	const freal ptotal;
	const freal ttotal;
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
};

/// Simply sets the ghost state as the given free-stream state
template <typename scalar, typename j_real = freal>
class Farfield : public FlowBC<scalar, j_real>
{
public:
	/// Setup farfield BC
	/** \sa FlowBC::FlowBC
	 * \param ufar Far-field state
	 */
	Farfield(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	         const std::array<freal,NVARS>& u_far);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	const std::array<freal,NVARS> uinf;
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
};

/// Simply sets the ghost state as the interior state
template <typename scalar, typename j_real = freal>
class Extrapolation : public FlowBC<scalar, j_real>
{
public:
	/// Setup extrapolation BC
	/** \sa FlowBC::FlowBC
	 */
	Extrapolation(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
};

/// Slip wall BC for Euler equations
template <typename scalar, typename j_real = freal>
class Slipwall : public FlowBC<scalar, j_real>
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
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
};

/// No-slip adiabatic wall BC for 2D NS equations
template <typename scalar, typename j_real = freal>
class Adiabaticwall2D : public FlowBC<scalar, j_real>
{
	static_assert(NDIM == 2, "Adiabaticwall2D is only defined for 2D cases.");
public:
	/// Setup adiabatic no-slip wall BC
	/** \sa FlowBC::FlowBC
	 */
	Adiabaticwall2D(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	                const freal wall_tangential_velocity);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
	const freal tangvel;
};

/// General adiabatic wall suitable for geometry in Cartesian coordinates
template <typename scalar, typename j_real = freal>
class Adiabaticwall : public FlowBC<scalar,j_real>
{
public:
	/// Setup adiabatic no-slip wall BC
	/** \sa FlowBC::FlowBC
	 */
	Adiabaticwall(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	              const std::array<freal,NDIM> wall_velocity);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
	const std::array<freal,NDIM> wallvel;
};

/// No-slip isothermal wall BC for 2D NS equations
template <typename scalar, typename j_real = freal>
class Isothermalwall2D : public FlowBC<scalar,j_real>
{
	static_assert(NDIM == 2, "Isothermalwall2D is only defined for 2D cases.");
public:
	/// Setup isothermal no-slip wall BC
	/** \sa FlowBC::FlowBC
	 */
	Isothermalwall2D(const int boundary_tag, const IdealGasPhysics<scalar>& gasphysics,
	                 const freal wall_tangential_velocity, const freal wall_temperature);

	/// Computes the ghost state given the interior state and normal vector
	void computeGhostState(const scalar *const uin, const scalar *const n,
	                       scalar *const __restrict ughost) const;

	/// Computes the Jacobian of the ghost state w.r.t. the interior state
	void computeGhostStateAndJacobian(const j_real *const uin, const j_real *const n,
	                                  j_real *const __restrict ug,
	                                  j_real *const __restrict dugdui) const;

protected:
	using FlowBC<scalar,j_real>::btag;
	using FlowBC<scalar,j_real>::phy;
	using FlowBC<scalar,j_real>::jphy;
	const freal tangvel;
	const freal walltemperature;
};

/// Create a set of pointers to immutable boundary condition objects, possibly of different types
/** The BC objects are mapped to their corresponding boundary markers specified in the control file.
 * \param conf Boundary condition parameters read from control file
 * \param physics Gas properties
 * \param uinf Free-stream state in conserved variables
 */
template <typename scalar>
std::map<int,const FlowBC<scalar>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<scalar>& physics,
                                                         const std::array<freal,NVARS>& uinf);

}
#endif

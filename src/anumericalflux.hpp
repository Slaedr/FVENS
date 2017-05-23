/** \file anumericalflux.hpp
 * \brief Numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#ifndef __ANUMERICALFLUX_H

#ifndef __ACONSTANTS_H
#include "aconstants.hpp"
#endif

#ifndef __AEULERFLUX_H
#include "aeulerflux.hpp"
#endif

#define __ANUMERICALFLUX_H 1

namespace acfd {

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, the numerical flux is computed.
 */
class InviscidFlux
{
protected:
	a_real g;						///< Adiabatic index
	const EulerFlux *const aflux;		///< Analytical flux context

public:
	/// Sets up data for the inviscid flux scheme
	InviscidFlux(const a_real gamma, const EulerFlux *const analyticalflux);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const flux) = 0;

	/// Computes the Jacobian of the inviscid flux across a face w.r.t. both left and right states
	/** dfdl is the `lower' block formed by the coupling between the elements adjoining the face,
	 * while dfdr is the `upper' block.
	 * The negative of the lower block is the contribution to the diagonal block of the left cell, and
	 * the negative of the upper block is the contribution to the diagonal block of the right cell.
	 */
	virtual void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const dfdl, a_real *const dfdr) = 0;

	virtual ~InviscidFlux();
};

class LocalLaxFriedrichsFlux : public InviscidFlux
{
public:
	LocalLaxFriedrichsFlux(const a_real gamma, const EulerFlux *const analyticalflux);
	void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const flux);
	void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const dfdl, a_real *const dfdr);
};

/// Given left and right states at each face, the Van-Leer flux-vector-splitting is calculated at each face
class VanLeerFlux : public InviscidFlux
{
public:
	VanLeerFlux(a_real gamma, const EulerFlux *const analyticalflux);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr);
};

/// Roe flux-difference splitting Riemann solver for the Euler equations
class RoeFlux : public InviscidFlux
{
public:
	RoeFlux(a_real gamma, const EulerFlux *const analyticalflux);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr);
};

/// Harten Lax Van-Leer numerical flux
class HLLFlux : public InviscidFlux
{
	/// Computes the Jacobian of the numerical flux w.r.t. left state
	void getFluxJac_left(const a_real *const ul, const a_real *const ur, const a_real *const n, a_real *const flux, a_real *const fluxd);
	/// Computes the Jacobian of the numerical flux w.r.t. right state
	void getFluxJac_right(const a_real *const ul, const a_real *const ur, const a_real *const n, a_real *const flux, a_real *const fluxd);

public:
	HLLFlux(a_real gamma, const EulerFlux *const analyticalflux);
	
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
	
	void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const dfdl, a_real *const dfdr);
	
	void get_frozen_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr);

	/// Computes both the flux and the 2 flux Jacobians
	void get_flux_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux, a_real *const dfdl, a_real *const dfdr);
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** From Remaki et. al., "Aerodynamic computations using FVM and HLLC".
 */
class HLLCFlux : public InviscidFlux
{
public:
	HLLCFlux(a_real gamma, const EulerFlux *const analyticalflux);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const dfdl, a_real *const dfdr);
};

} // end namespace acfd

#endif

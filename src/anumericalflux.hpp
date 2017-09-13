/** \file anumericalflux.hpp
 * \brief Numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#ifndef __ANUMERICALFLUX_H

#ifndef __ACONSTANTS_H
#include "aconstants.hpp"
#endif

#ifndef __APHYSICS_H
#include "aphysics.hpp"
#endif

#define __ANUMERICALFLUX_H 1

namespace acfd {

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, 
 * the numerical flux and its Jacobian w.r.t. left and right states is computed.
 */
class InviscidFlux
{
protected:
	const IdealGasPhysics *const physics;		///< Analytical flux context
	a_real g;								///< Adiabatic index

public:
	/// Sets up data for the inviscid flux scheme
	InviscidFlux(const IdealGasPhysics *const analyticalflux);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const a_real *const uleft, const a_real *const uright, 
			const a_real* const n, 
			a_real *const flux) = 0;

	/// Computes the Jacobian of inviscid flux across a face w.r.t. both left and right states
	/** dfdl is the `lower' block formed by the coupling between elements adjoining the face,
	 * while dfdr is the `upper' block.
	 * The negative of the lower block is the contribution to diagonal block of left cell, and
	 * the negative of the upper block is the contribution to diagonal block of right cell.
	 */
	virtual void get_jacobian(const a_real *const uleft, const a_real *const uright, 
			const a_real* const n,
			a_real *const dfdl, a_real *const dfdr) = 0;

	virtual ~InviscidFlux();
};

class LocalLaxFriedrichsFlux : public InviscidFlux
{
public:
	LocalLaxFriedrichsFlux(const IdealGasPhysics *const analyticalflux);
	void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, 
			a_real *const flux);
	void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Van-Leer flux-vector-splitting
class VanLeerFlux : public InviscidFlux
{
public:
	VanLeerFlux(const IdealGasPhysics *const analyticalflux);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Roe flux-difference splitting Riemann solver for the Euler equations
class RoeFlux : public InviscidFlux
{
public:
	RoeFlux(const IdealGasPhysics *const analyticalflux);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
};

/// Harten Lax Van-Leer numerical flux
class HLLFlux : public InviscidFlux
{
	/// Computes the Jacobian of the numerical flux w.r.t. left state
	void getFluxJac_left(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);
	/// Computes the Jacobian of the numerical flux w.r.t. right state
	void getFluxJac_right(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);

public:
	HLLFlux(const IdealGasPhysics *const analyticalflux);
	
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
	
	void get_frozen_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);

	/// Computes both the flux and the 2 flux Jacobians
	void get_flux_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux, a_real *const dfdl, a_real *const dfdr);
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** Implemented as described by Remaki et al. \cite invflux_remaki
 */
class HLLCFlux : public InviscidFlux
{
	/// Computes the Jacobian of the numerical flux w.r.t. left state
	void getFluxJac_left(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);
	/// Computes the Jacobian of the numerical flux w.r.t. right state
	void getFluxJac_right(const a_real *const ul, const a_real *const ur, const a_real *const n, 
			a_real *const flux, a_real *const fluxd);
public:
	HLLCFlux(const IdealGasPhysics *const analyticalflux);
	
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux);
	
	void get_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const dfdl, a_real *const dfdr);
	
	/// Computes both the flux and the 2 flux Jacobians
	void get_flux_jacobian(const a_real *const ul, const a_real *const ur, const a_real* const n, 
			a_real *const flux, a_real *const dfdl, a_real *const dfdr);
};

} // end namespace acfd

#endif

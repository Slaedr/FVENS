/** \file anumericalflux.hpp
 * \brief Numerical flux schemes for Euler and Navier-Stokes equations.
 * \author Aditya Kashi
 * \date March 2015
 */

#ifndef __ANUMERICALFLUX_H

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX_H
#include <amatrix.hpp>
#endif

#define __ANUMERICALFLUX_H 1

namespace acfd {

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, the numerical flux is computed.
 */
class InviscidFlux
{
protected:
	a_real g;	///< Adiabatic index

public:
	/// Sets up data for the inviscid flux scheme
	InviscidFlux(a_real gamma);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const flux) = 0;

	virtual void get_jacobian(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const dfdl, a_real *const dfdr);

	virtual ~InviscidFlux();
};

class LocalLaxFriedrichsFlux : public InviscidFlux
{
public:
	LocalLaxFriedrichsFlux(const a_real gamma);
	void get_flux(const a_real *const uleft, const a_real *const uright, const a_real* const n, a_real *const flux);
};

/// Given left and right states at each face, the Van-Leer flux-vector-splitting is calculated at each face
class VanLeerFlux : public InviscidFlux
{
	amat::Matrix<a_real> fiplus;
	amat::Matrix<a_real> fjminus;

public:
	VanLeerFlux(a_real gamma);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
};

/// Roe flux-difference splitting Riemann solver for the Euler equations
class RoeFlux : public InviscidFlux
{
public:
	RoeFlux(a_real gamma);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** From Remaki et. al., "Aerodynamic computations using FVM and HLLC".
 */
class HLLCFlux : public InviscidFlux
{
	amat::Matrix<a_real> utemp;
public:
	HLLCFlux(a_real gamma);
	void get_flux(const a_real *const ul, const a_real *const ur, const a_real* const n, a_real *const flux);
};

} // end namespace acfd

#endif

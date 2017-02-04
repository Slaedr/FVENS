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

/// Adiabatic index
const double g = 1.4;

/// Abstract class from which to derive all numerical flux classes
/** The class is such that given the left and right states and a face normal, the numerical flux is computed.
 */
class InviscidFlux
{
protected:
	int nvars;		///< Number of conserved variables
	int ndim;		///< Number of spatial dimensions involved
	acfd_real g;	///< Adiabatic index

public:
	/// Sets up data for the inviscid flux scheme
	InviscidFlux(int num_vars, int num_dims, acfd_real gamma);

	/** Computes flux across a face with
	 * \param[in] uleft is the vector of left states for the face
	 * \param[in] uright is the vector of right states for the face
	 * \param[in] n is the normal vector to the face
	 * \param[in|out] flux contains the computed flux
	 */
	virtual void get_flux(const acfd_real *const uleft, const acfd_real *const uright, const acfd_real* const n, acfd_real *const flux) = 0;
	virtual ~InviscidFlux();
};

/// Given left and right states at each face, the Van-Leer flux-vector-splitting is calculated at each face
class VanLeerFlux : public InviscidFlux
{
	amat::Matrix<acfd_real> fiplus;
	amat::Matrix<acfd_real> fjminus;

public:
	VanLeerFlux(int num_vars, int num_dims, acfd_real gamma);
	void get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux);
};

/// Roe flux-difference splitting Riemann solver for the Euler equations
class RoeFlux : public InviscidFlux
{
public:
	RoeFlux(int num_vars, int num_dims, acfd_real gamma);
	void get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux);
};

/// Harten Lax Van-Leer numerical flux with contact restoration by Toro
/** From Remaki et. al., "Aerodynamic computations using FVM and HLLC".
 */
class HLLCFlux : public InviscidFlux
{
	amat::Matrix<acfd_real> utemp;
public:
	HLLCFlux(int num_vars, int num_dims, acfd_real gamma);
	void get_flux(const acfd_real *const ul, const acfd_real *const ur, const acfd_real* const n, acfd_real *const flux);
};

} // end namespace acfd

#endif

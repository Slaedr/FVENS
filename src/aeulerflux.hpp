/** \file aeulerflux.hpp
 * \brief Provides analytical flux computation contexts
 * \author Aditya Kashi
 * \date 2017 May 12
 */

#ifndef __AEULERFLUX_H
#define __AEULERFLUX_H

namespace acfd {

/// Abstract class providing analytical fluxes and their Jacobians
class Flux
{
public:
	/// Computes the flux vector along some direction
	virtual void evaluate_normal_flux(const a_real *const u, const a_real* const n, a_real *const flux) const = 0;

	/// Computes the Jacobian of the flux along some direction, at the given state
	virtual void evaluate_normal_jacobian(const a_real *const u, const a_real* const n, a_real *const dfdu) const = 0;
};
	
/// Computation of the single-phase ideal gas Euler flux corresponding to any given state and along any given face-normal
class EulerFlux : public Flux
{
protected:
	const a_real g;
public:
	EulerFlux(a_real _g) : g(_g)
	{ }

	void evaluate_normal_flux(const a_real *const __restrict__ u, const a_real* const __restrict__ n, a_real *const __restrict__ flux) const
	{
		a_real vn = (u[1]*n[0] + u[2]*n[1])/u[0];
		a_real p = (g-1.0)*(u[3] - 0.5*(u[1]*u[1] + u[2]*u[2])/u[0]);
		flux[0] = u[0] * vn;
		flux[1] = vn*u[1] + p*n[0];
		flux[2] = vn*u[2] + p*n[1];
		flux[3] = vn*(u[3] + p);
	}
	
	/// Computes the Jacobian of the flux along some direction, at the given state
	/** The flux Jacobian matrix dfdu is assumed stored in a row-major 1-dimensional array.
	 */
	void evaluate_normal_jacobian(const a_real *const __restrict__ u, const a_real* const __restrict__ n, a_real *const __restrict__ dfdu) const
	{
		a_real rhovn = u[1]*n[0]+u[2]*n[1], u02 = u[0]*u[0];
		// first row
		dfdu[0] = 0; 
		dfdu[1] = n[0]; 
		dfdu[2] = n[1]; 
		dfdu[3] = 0;
		// second row
		dfdu[4] = (-rhovn*u[1] + (g-1)*n[0]*(u[1]*u[1]+u[2]*u[2])/2.0) / u02;
		dfdu[5] = ((3.0-g)*u[1]*n[0] + u[2]*n[1]) / u[0];
		dfdu[6] = (n[1]*u[1]-(g-1)*n[0]*u[2])/u[0];
		dfdu[7] = (g-1)*n[0];
		// third row
		dfdu[8] = (-rhovn*u[2] + (g-1)*n[1]*(u[1]*u[1]+u[2]*u[2])/2.0) / u02;
		dfdu[9] = (n[0]*u[2]-(g-1)*n[1]*u[1])/u[0];
		dfdu[10]= ((3.0-g)*u[2]*n[1]+u[1]*n[0])/u[0];
		dfdu[11]= (g-1)*n[1];
		// fourth row
		dfdu[12]= rhovn*((g-1) * (u[1]*u[1]+u[2]*u[2])/(u02*u[0]) - g*u[3]/u02);
		dfdu[13]= g*u[3]*n[0]/u[0] - (g-1)*0.5/u02*(3*u[1]*u[1]*n[0]+u[2]*u[2]*n[0]+2*u[1]*u[2]*n[1]);
		dfdu[14]= g*u[3]*n[1]/u[0] - (g-1)*0.5/u02*(2*u[1]*u[2]*n[0]+u[1]*u[1]*n[1]+3*u[2]*u[2]*n[1]);
		dfdu[15]= g*rhovn/u[0];
	}
};

}
#endif

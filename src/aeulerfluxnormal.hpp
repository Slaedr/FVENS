#ifndef __AEULERFLUXNORMAL_H

#define __AEULERFLUXNORMAL_H
#ifndef __AMATRTIX_H
#include <amatrix.hpp>
#endif

namespace acfd {

/// Abstract class providing analytical fluxes and their Jacobians along some normal direction
class Flux
{
public:
	virtual void evaluate_normal_flux(const a_real *const u, const a_real* const n, a_real *const flux) const = 0;
	virtual void evaluate_normal_jacobian(const a_real *const u, const a_real* const n, a_real *const dfdu) const = 0;
};
	
/// Computation of the single-phase ideal gas Euler flux corresponding to any given state and along any given face-normal
class EulerFlux : public Flux
{
protected:
	const a_real g;
public:
	FluxFunction (a_real _g) : g(_g)
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
	
	void evaluate_normal_jacobian(const a_real *const __restrict__ u, const a_real* const __restrict__ n, a_real *const __restrict__ dfdu) const
	{
		a_real rhovn = u[1]*n[0]+u[2]*n[1], u02 = u[0]*u[0];
		// first row
		dfdu[0] = 0; dfdu[1] = n[0]; dfdu[2] = n[1]; dfdu[3] = 0;
		// second row
		dfdu[4] = (-rhovn*u[1] + (g-1)*n[0]*(u[1]*u[1]+u[2]*u[2])/2.0)/u02;
		dfdu[5] = n[0]*u[1]/u[0]*(2-g);
		dfdu[6] = (n[1]*u[1]-(g-1)*n[0]*u[2])/u[0];
		dfdu[7] = (g-1)*n[0];
		// third row
		dfdu[8] = (-rhovn*u[2] + (g-1)*n[1]*(u[1]*u[1]+u[2]*u[2])/2.0)/u02;
	}
	
	void evaluate_flux_2(const amat::Matrix<a_real>& state, const int ielem, const a_real* const n, amat::Matrix<a_real>& flux, const int iside) const
	{
		a_real vn = (state.get(ielem,1)*n[0] + state.get(ielem,2)*n[1])/state.get(ielem,0);
		a_real p = (g-1.0)*(state.get(ielem,3) - 0.5*(state.get(ielem,1)*state.get(ielem,1) + state.get(ielem,2)*state.get(ielem,2))/state.get(ielem,0));
		flux(iside,0) = state.get(ielem,0) * vn;
		flux(iside,1) = vn*state.get(ielem,1) + p*n[0];
		flux(iside,2) = vn*state.get(ielem,2) + p*n[1];
		flux(iside,3) = vn*(state.get(ielem,3) + p);
	}
};

}
#endif

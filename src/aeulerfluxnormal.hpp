#ifndef __AEULERFLUXNORMAL_H

#define __AEULERFLUXNORMAL_H
#ifndef __AMATRTIX_H
#include <amatrix.hpp>
#endif

namespace acfd {

/// Computation of the single-phase ideal gas Euler flux corresponding to any given state and along any given face-normal
class FluxFunction
{
protected:
	const acfd_real gamma;
public:
	FluxFunction (acfd_real _gamma) : gamma(_gamma)
	{ }

	void evaluate_flux(const amat::Matrix<acfd_real>& state, const acfd_real* const n, amat::Matrix<acfd_real>& flux) const
	{
		acfd_real vn = (state.get(1)*n[0] + state.get(2)*n[1])/state.get(0);
		acfd_real p = (gamma-1.0)*(state.get(3) - 0.5*(state.get(1)*state.get(1) + state.get(2)*state.get(2))/state.get(0));
		flux(0) = state.get(0) * vn;
		flux(1) = vn*state.get(1) + p*n[0];
		flux(2) = vn*state.get(2) + p*n[1];
		flux(3) = vn*(state.get(3) + p);
	}
};

}
#endif

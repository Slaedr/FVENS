/** \file abc.cpp
 * \brief Boundary conditions management
 * \author Aditya Kashi
 * \date 2018-05
 */

#include "abc.hpp"

namespace fvens {

template <typename scalar>
FlowBC<scalar>::FlowBC(const int bc_tag, const IdealGasPhysics& gasphysics)
	: btag{bc_tag}, phy{gasphysics}
{ }

template <typename scalar>
SlipWall<scalar>::SlipWall(const int bc_tag, const IdealGasPhysics& gasphysics)
	: FlowBC<scalar>(bc_tag, gasphysics)
{ }

template <typename scalar>
void SlipWall<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                         scalar *const gs) const
{
	const a_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	gs[0] = ins[0];
	for(int i = 1; i < NDIM+1; i++)
		gs[i] = ins[i] - 2.0*vni*n[i-1]*ins[0];
	gs[3] = ins[3];
}

template <typename scalar>
void SlipWall<scalar>::computeJacobian(const scalar *const ins, const scalar *const n,
                                       scalar *const gs,
                                       scalar *const dgs) const
{
	const a_real vni = dimDotProduct(&ins[1],n)/ins[0];
	a_real dvni[NVARS];
	dvni[0] = -vni/ins[0];
	for(int i = 1; i < NDIM+1; i++)
		dvni[i] = n[i-1]/ins[0];
	dvni[NDIM+1] = 0;

	gs[0] = ins[0];
		
	dgs[0] = 1.0;

	gs[1] = ins[1] - 2.0*vni*n[0]*ins[0];
		
	dgs[NVARS+0] = -2.0*n[0]*(dvni[0]*ins[0]+vni);
	dgs[NVARS+1] = 1.0 - 2.0*n[0]*ins[0]*dvni[1];
	dgs[NVARS+2] = -2.0*n[0]*ins[0]*dvni[2];

	gs[2] = ins[2] - 2.0*vni*n[1]*ins[0];
		
	dgs[2*NVARS+0] = -2.0*n[1]*(dvni[0]*ins[0]+vni);
	dgs[2*NVARS+1] = -2.0*n[1]*ins[0]*dvni[1];
	dgs[2*NVARS+2] = 1.0 - 2.0*n[1]*ins[0]*dvni[2];

	gs[3] = ins[3];
		
	dgs[3*NVARS+3] = 1.0;
}

template <typename scalar>
InOutFlow<scalar>::InOutFlow(const int bc_tag, const IdealGasPhysics& gasphysics,
                             const std::array<scalar,NDIM>& u_far)
	: FlowBC<scalar>(bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar>
void InOutFlow<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                         scalar *const gs) const
{
	const a_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	const a_real ci = phy.getSoundSpeedFromConserved(ins);
	const a_real Mni = vni/ci;
	const a_real pinf = phy.getFreestreamPressure();

	/* At inflow, ghost cell state is determined by farfield state; the Riemann solver
	 * takes care of signal propagation at the boundary.
	 */
	if(Mni <= 0)
		{
			for(int i = 0; i < NVARS; i++)
				gs[i] = uinf[i];
		}

	/* At subsonic outflow, pressure is taken from farfield, the other 3 quantities
	 * are taken from the interior.
	 */
	else if(Mni < 1)
		{
			gs[0] = ins[0];
			for(int i = 1; i < NDIM+1; i++)
				gs[i] = ins[i];
			//gs[3] = pinf/(physics.g-1.0) + 0.5*(ins[1]*ins[1]+ins[2]*ins[2])/ins[0];
			gs[NDIM+1] = phy.getEnergyFromPressure(pinf, ins[0],
			                                       dimDotProduct(&ins[1],&ins[1])/(ins[0]*ins[0]) );
		}
		
	// At supersonic outflow, everything is taken from the interior
	else
		{
			for(int i = 0; i < NVARS; i++)
				gs[i] = ins[i];
		}
}

template <typename scalar>
void InOutFlow<scalar>::computeJacobian(const scalar *const ins, const scalar *const n,
                                        scalar *const gs,
                                        scalar *const dgs) const
{
	const a_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	const a_real ci = phy.getSoundSpeedFromConserved(ins);
	const a_real Mni = vni/ci;

	const a_real pinf = phy.getPressureFromConserved(&uinf[0]);

	/* At inflow, ghost cell state is determined by farfield state; the Riemann solver
	 * takes care of signal propagation at the boundary.
	 */
	if(Mni <= 0)
	{
		for(int i = 0; i < NVARS; i++)
			gs[i] = uinf[i];
	}

	/* At subsonic outflow, pressure is taken from farfield, the other 3 quantities
	 * are taken from the interior.
	 */
	else if(Mni <= 1)
	{
		gs[0] = ins[0];
		gs[1] = ins[1];
		gs[2] = ins[2];
		for(int k = 0; k < NVARS-1; k++)
			dgs[k*NVARS+k] = 1.0;

		//gs[3] = pinf/(physics.g-1.0) + 0.5*(ins[1]*ins[1]+ins[2]*ins[2])/ins[0];
		gs[NDIM+1] = phy.getEnergyFromPressure
			( pinf, ins[0], dimDotProduct(&ins[1],&ins[1])/(ins[0]*ins[0]) );

		dgs[3*NVARS+0] = -0.5*(ins[1]*ins[1]+ins[2]*ins[2])/(ins[0]*ins[0]);
		dgs[3*NVARS+1] = ins[1]/ins[0];
		dgs[3*NVARS+2] = ins[2]/ins[0];
		dgs[3*NVARS+3] = 0;
	}

	// At supersonic outflow, everything is taken from the interior
	else
	{
		for(int i = 0; i < NVARS; i++) {
			gs[i] = ins[i];
			dgs[i*NVARS+i] = 1.0;
		}
	}
}

template <typename scalar>
InFlow<scalar>::InFlow(const int bc_tag, const IdealGasPhysics& gasphysics,
                       const scalar t_pressure, const scalar t_temp)
	: FlowBC<scalar>(bc_tag, gasphysics), ptotal{t_pressure}, ttotal{t_temp}
{ }

template <typename scalar>
void InFlow<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                       scalar *const gs) const
{
}

template <typename scalar>
void InFlow<scalar>::computeJacobian(const scalar *const ins, const scalar *const n,
                                     scalar *const gs,
                                     scalar *const dgs) const
{
}

template <typename scalar>
Farfield<scalar>::Farfield(const int bc_tag, const IdealGasPhysics& gasphysics,
                           const std::array<scalar,NDIM>& u_far)
	: FlowBC<scalar>(bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar>
void Farfield<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                         scalar *const gs) const
{
	for(int i = 0; i < NVARS; i++)
		gs[i] = uinf[i];
}

template <typename scalar>
void Farfield<scalar>::computeJacobian(const scalar *const ins, const scalar *const n,
                                       scalar *const gs,
                                       scalar *const dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;
}

}

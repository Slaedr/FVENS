/** \file abc.cpp
 * \brief Boundary conditions management
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

#include <iostream>
#include "abc.hpp"
#include "physics/aphysics_defs.hpp"
#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template <typename scalar, typename j_real>
FlowBC<scalar,j_real>::FlowBC(const BCType btype, const int bc_tag,
                              const IdealGasPhysics<scalar>& gasphysics)
	: bctype{btype}, btag{bc_tag}, phy{gasphysics},
	  jphy(gasphysics.g, gasphysics.Minf, gasphysics.Tinf, gasphysics.Reinf, gasphysics.Pr)
{ }

template <typename scalar, typename j_real>
FlowBC<scalar,j_real>::~FlowBC()
{ }

template <typename scalar, typename j_real>
InOutFlow<scalar,j_real>::InOutFlow(const int bc_tag,
                                    const IdealGasPhysics<scalar>& gasphysics,
                                    const std::array<freal,NVARS>& u_far)
	: FlowBC<scalar,j_real>(INFLOW_OUTFLOW_BC, bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar, typename j_real>
void InOutFlow<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                 scalar *const __restrict gs) const
{
	const scalar vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	const scalar ci = phy.getSoundSpeedFromConserved(ins);
	const scalar Mni = vni/ci;
	const scalar pinf = phy.getFreestreamPressure();

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

template <typename scalar, typename j_real>
void InOutFlow<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins,
                                                            const j_real *const n,
                                                            j_real *const __restrict gs,
                                                            j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const j_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	const j_real ci = jphy.getSoundSpeedFromConserved(ins);
	const j_real Mni = vni/ci;

	const j_real pinf = jphy.getPressureFromConserved(&uinf[0]);

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
		gs[NDIM+1] = jphy.getEnergyFromPressure
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

template <typename scalar, typename j_real>
InFlow<scalar,j_real>::InFlow(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                              const freal t_pressure, const freal t_temp)
	: FlowBC<scalar,j_real>(SUBSONIC_INFLOW_BC, bc_tag, gasphysics), ptotal{t_pressure}, ttotal{t_temp}
{ }

/** Assumes the flow at the boundary is isentropic. Uses the the fact that the stagnation speed
 * of sound and the outgoing Riemann invariant are conserved across the face.
 */
template <typename scalar, typename j_real>
void InFlow<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                              scalar *const __restrict gs) const
{
	const scalar ci = phy.getSoundSpeedFromConserved(ins);
	// Outgoing Riemann invariant
	const scalar Rminus = dimDotProduct(&ins[1],&n[0])/ins[0] - ci/(2*phy.g - 1.0);
	// Square of stagnation speed of sound
	const scalar co2 = ci*ci + (phy.g-1.0)/2.0 * dimDotProduct(&ins[1],&ins[1])/(ins[0]*ins[0]);

	const scalar q = sqrt((phy.g+1)*co2/((phy.g-1)*Rminus*Rminus) - (phy.g-1)/2.0);

	// Ghost state values
	const scalar cg = -Rminus*(phy.g-1)/(phy.g+1) * (1.0 + q);
	const scalar tg = ttotal*cg*cg/co2;
	const scalar pg = ptotal * pow(tg/ttotal, phy.g/(phy.g-1.0));
	gs[0] = phy.getDensityFromPressureTemperature(pg,tg);
	const scalar vgmag = sqrt(2.0/(phy.g-1.0)*(co2 - cg*cg));

	// Get velocity components of ghost state assuming it's (anti-)parallel to the face normal
	scalar vg[NDIM];
	getComponentsCartesian<scalar>(vgmag, n, vg);
	for(int i = 0; i < NDIM; i++)
		gs[i+1] = gs[0]*vg[i];
	gs[NDIM+1] = phy.getEnergyFromPressure(pg,gs[0],vgmag*vgmag);
}

template <typename scalar, typename j_real>
void InFlow<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins, const j_real *const n,
                                                         j_real *const __restrict gs,
                                                         j_real *const __restrict dgs) const
{
	std::cout << "Not implemented!\n";
	std::exit(-1);
}

template <typename scalar, typename j_real>
Farfield<scalar,j_real>::Farfield(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                                  const std::array<freal,NVARS>& u_far)
	: FlowBC<scalar,j_real>(FARFIELD_BC, bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar, typename j_real>
void Farfield<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                scalar *const __restrict gs) const
{
	for(int i = 0; i < NVARS; i++)
		gs[i] = uinf[i];
}

template <typename scalar, typename j_real>
void Farfield<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins, const j_real *const n,
                                                           j_real *const __restrict gs,
                                                           j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;
	for(int i = 0; i < NVARS; i++)
		gs[i] = uinf[i];
}

template <typename scalar, typename j_real>
Slipwall<scalar,j_real>::Slipwall(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics)
	: FlowBC<scalar,j_real>(SLIP_WALL_BC, bc_tag, gasphysics)
{ }

template <typename scalar, typename j_real>
void Slipwall<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                scalar *const __restrict gs) const
{
	const scalar vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	gs[0] = ins[0];
	for(int i = 1; i < NDIM+1; i++)
		gs[i] = ins[i] - 2.0*vni*n[i-1]*ins[0];
	gs[3] = ins[3];
}

/// \todo Make this dimension-independent
template <typename scalar, typename j_real>
void Slipwall<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins, const j_real *const n,
                                                           j_real *const __restrict gs,
                                                           j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const j_real vni = dimDotProduct(&ins[1],n)/ins[0];
	j_real dvni[NVARS];
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

template <typename scalar, typename j_real>
Adiabaticwall2D<scalar,j_real>::Adiabaticwall2D(const int bc_tag,
                                                const IdealGasPhysics<scalar>& gasphysics,
                                                const freal wall_tangential_velocity)
	: FlowBC<scalar,j_real>(ADIABATIC_WALL_BC, bc_tag, gasphysics), tangvel{wall_tangential_velocity}
{ }

template <typename scalar, typename j_real>
void Adiabaticwall2D<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                       scalar *const __restrict gs) const
{
	const scalar tangMomentum = tangvel * ins[0];
	gs[0] = ins[0];
	gs[1] =  2.0*tangMomentum*n[1] - ins[1];
	gs[2] = -2.0*tangMomentum*n[0] - ins[2];
	gs[3] = ins[3];
}

template <typename scalar, typename j_real>
void Adiabaticwall2D<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins,
                                                                  const j_real *const n,
                                                                  j_real *const __restrict gs,
                                                                  j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const j_real tangMomentum = tangvel * ins[0];
	gs[0] = ins[0];
	dgs[0] = 1.0;

	gs[1] =  2.0*tangMomentum*n[1] - ins[1];
	dgs[NVARS+0] = 2.0*tangvel*n[1];
	dgs[NVARS+1] = -1.0;

	gs[2] = -2.0*tangMomentum*n[0] - ins[2];
	dgs[2*NVARS+0] = -2.0*tangvel*n[0];
	dgs[2*NVARS+2] = -1.0;

	gs[3] = ins[3];
	dgs[3*NVARS+3] = 1.0;
}

template <typename scalar, typename j_real>
Adiabaticwall<scalar,j_real>::Adiabaticwall(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                                            const std::array<freal,NDIM> wall_velocity)
	: FlowBC<scalar,j_real>(ADIABATIC_WALL_BC, bc_tag, gasphysics), wallvel(wall_velocity)
{ }

template <typename scalar, typename j_real>
void Adiabaticwall<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                     scalar *const __restrict gs) const
{
	gs[0] = ins[0];
	for(int i= 1; i< NDIM+1; i++)
		gs[i] =  2.0*ins[0]*wallvel[i] - ins[i];
	gs[NDIM+1] = ins[NDIM+1];
}

template <typename scalar, typename j_real>
void Adiabaticwall<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins,
                                                                const j_real *const n,
                                                                j_real *const __restrict gs,
                                                                j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	gs[0] = ins[0];
	dgs[0] = 1.0;

	for(int i= 1; i< NDIM+1; i++) {
		gs[i] =  2.0*ins[0]*wallvel[i] - ins[i];

		dgs[i*NVARS+0] = 2.0*wallvel[i];
		dgs[i*NVARS+i] = -1.0;
	}

	gs[NVARS-1] = ins[NVARS-1];
	dgs[NVARS*NVARS-1] = 1.0;
}

template <typename scalar, typename j_real>
Isothermalwall2D<scalar,j_real>::Isothermalwall2D(const int bc_tag,
                                                  const IdealGasPhysics<scalar>& gasphysics,
                                                  const freal wtv, const freal temp)
	: FlowBC<scalar,j_real>(ISOTHERMAL_WALL_BC, bc_tag, gasphysics), tangvel{wtv}, walltemperature{temp}
{ }

template <typename scalar, typename j_real>
void Isothermalwall2D<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                        scalar *const __restrict gs) const
{
	// pressure in the interior cell
	const scalar p = phy.getPressureFromConserved(ins);

	// temperature in the ghost cell
	const scalar gtemp = 2.0*walltemperature - phy.getTemperature(ins[0],p);

	//gs[0] = physics.getDensityFromPressureTemperature(p, gtemp);
	gs[0] = ins[0];
	gs[1] = gs[0]*( 2.0*tangvel*n[1] - ins[1]/ins[0]);
	gs[2] = gs[0]*(-2.0*tangvel*n[0] - ins[2]/ins[0]);
	const scalar vmag2 = dimDotProduct(&gs[1],&gs[1])/(gs[0]*gs[0]);
	gs[3] = phy.getEnergyFromTemperature(gtemp, gs[0], vmag2);
}

template <typename scalar, typename j_real>
void Isothermalwall2D<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins,
                                                                   const j_real *const n,
                                                                   j_real *const __restrict gs,
                                                                   j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	/// \todo Fix isothermal BC Jacobian
	// FIXME: Wrong Jacobian
	const j_real tangMomentum = tangvel * ins[0];
	gs[0] = ins[0];
	dgs[0] = 1.0;

	gs[1] =  2.0*tangMomentum*n[1] - ins[1];
	dgs[NVARS+0] = 2.0*tangvel*n[1];
	dgs[NVARS+1] = -1.0;

	gs[2] = -2.0*tangMomentum*n[0] - ins[2];
	dgs[2*NVARS+0] = -2.0*tangvel*n[0];
	dgs[2*NVARS+2] = -1.0;

	const j_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]);
	gs[3] = jphy.getEnergyFromTemperature(walltemperature, ins[0], vmag2);

	j_real dvmag2[NVARS]; // derivative of vmag2 w.r.t. ins
	dvmag2[0] = -2.0*dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]*ins[0]);
	for(int i = 1; i < NDIM+1; i++)
		dvmag2[i] = 1.0/(ins[0]*ins[0]) * gs[i] * (-1.0);
	dvmag2[NDIM+1] = 0;

	// dummy dT for the next function call
	j_real dT[NVARS];
	for(int i = 0; i < NVARS; i++) dT[i] = 0;

	jphy.getJacobianEnergyFromJacobiansTemperatureVmag2(walltemperature, ins[0], vmag2,
	                                                    dT, dvmag2, &dgs[3*NVARS]);
}

template <typename scalar, typename j_real>
Extrapolation<scalar,j_real>::Extrapolation(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics)
	: FlowBC<scalar,j_real>(EXTRAPOLATION_BC, bc_tag, gasphysics)
{ }

template <typename scalar, typename j_real>
void Extrapolation<scalar,j_real>::computeGhostState(const scalar *const ins, const scalar *const n,
                                                     scalar *const __restrict gs) const
{
	for(int k = 0; k < NVARS; k++) {
		gs[k] = ins[k];
	}
}

template <typename scalar, typename j_real>
void Extrapolation<scalar,j_real>::computeGhostStateAndJacobian(const j_real *const ins,
                                                                const j_real *const n,
                                                                j_real *const __restrict gs,
                                                                j_real *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;
	for(int k = 0; k < NVARS; k++) {
		gs[k] = ins[k];
		dgs[k*NVARS+k] = 1.0;
	}
}

template class InOutFlow<freal>;
template class Slipwall<freal>;
template class Adiabaticwall2D<freal>;
template class Adiabaticwall<freal>;
template class Isothermalwall2D<freal>;
template class Extrapolation<freal>;
template class InFlow<freal>;
template class Farfield<freal>;

#ifdef USE_ADOLC
template class InOutFlow<adouble>;
template class Slipwall<adouble>;
template class Adiabaticwall2D<adouble>;
template class Adiabaticwall<adouble>;
template class Isothermalwall2D<adouble>;
template class Extrapolation<adouble>;
template class InFlow<adouble>;
template class Farfield<adouble>;
#endif


template <typename scalar>
std::map<int,const FlowBC<scalar>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<scalar>& physics,
                                                         const std::array<freal,NVARS>& uinf)
{
	std::map<int,const FlowBC<scalar>*> bcmap;

	for(auto itbc = conf.begin(); itbc != conf.end(); itbc++) {
		const FlowBC<scalar>* bc;

		switch(itbc->bc_type) {
		case SLIP_WALL_BC:
			bc = new Slipwall<scalar>(itbc->bc_tag, physics);
			break;
		case FARFIELD_BC:
			bc = new Farfield<scalar>(itbc->bc_tag, physics, uinf);
			break;
		case INFLOW_OUTFLOW_BC:
			bc = new InOutFlow<scalar>(itbc->bc_tag, physics, uinf);
			break;
		case SUBSONIC_INFLOW_BC:
			bc = new InFlow<scalar>(itbc->bc_tag, physics, itbc->bc_vals[0], itbc->bc_vals[1]);
			break;
		case EXTRAPOLATION_BC:
			bc = new Extrapolation<scalar>(itbc->bc_tag, physics);
			break;
		case ADIABATIC_WALL_BC:
			bc = new Adiabaticwall2D<scalar>(itbc->bc_tag, physics, itbc->bc_vals[0]);
			break;
		case ISOTHERMAL_WALL_BC:
			bc = new Isothermalwall2D<scalar>(itbc->bc_tag, physics, itbc->bc_vals[0],
			                                  itbc->bc_vals[1]);
			break;
		default:
			throw std::runtime_error("BC type not implemented yet!");
		}

		bcmap[itbc->bc_tag] = bc;
	}
	return bcmap;
}

template
std::map<int,const FlowBC<freal>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<freal>& physics,
                                                         const std::array<freal,NVARS>& uinf);

#ifdef USE_ADOLC
template
std::map<int,const FlowBC<adouble>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<adouble>& physics,
                                                         const std::array<freal,NVARS>& uinf);
#endif
}

/** \file abc.cpp
 * \brief Boundary conditions management
 * \author Aditya Kashi
 * \date 2018-05
 */

#include <iostream>
#include "abc.hpp"

namespace fvens {

template <typename scalar>
FlowBC<scalar>::FlowBC(const BCType btype, const int bc_tag,
                       const IdealGasPhysics<scalar>& gasphysics)
	: bctype{btype}, btag{bc_tag}, phy{gasphysics}
{ }

template <typename scalar>
FlowBC<scalar>::~FlowBC()
{ }

template <typename scalar>
InOutFlow<scalar>::InOutFlow(const int bc_tag,
                             const IdealGasPhysics<scalar>& gasphysics,
                             const std::array<scalar,NVARS>& u_far)
	: FlowBC<scalar>(INFLOW_OUTFLOW_BC, bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar>
void InOutFlow<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
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
void InOutFlow<scalar>::computeGhostStateAndJacobian(const scalar *const ins, const scalar *const n,
                                                     scalar *const __restrict gs,
                                                     scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

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
InFlow<scalar>::InFlow(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                       const scalar t_pressure, const scalar t_temp)
	: FlowBC<scalar>(SUBSONIC_INFLOW_BC, bc_tag, gasphysics), ptotal{t_pressure}, ttotal{t_temp}
{ }

/** Assumes the flow at the boundary is isentropic. Uses the the fact that the stagnation speed
 * of sound and the outgoing Riemann invariant are conserved across the face.
 */
template <typename scalar>
void InFlow<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
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

template <typename scalar>
void InFlow<scalar>::computeGhostStateAndJacobian(const scalar *const ins, const scalar *const n,
                                                  scalar *const __restrict gs,
                                                  scalar *const __restrict dgs) const
{
	std::cout << "Not implemented!\n";
	std::exit(-1);
}

template <typename scalar>
Farfield<scalar>::Farfield(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                           const std::array<scalar,NVARS>& u_far)
	: FlowBC<scalar>(FARFIELD_BC, bc_tag, gasphysics), uinf(u_far)
{ }

template <typename scalar>
void Farfield<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                         scalar *const __restrict gs) const
{
	for(int i = 0; i < NVARS; i++)
		gs[i] = uinf[i];
}

template <typename scalar>
void Farfield<scalar>::computeGhostStateAndJacobian(const scalar *const ins, const scalar *const n,
                                                    scalar *const __restrict gs,
                                                    scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;
	for(int i = 0; i < NVARS; i++)
		gs[i] = uinf[i];
}

template <typename scalar>
Slipwall<scalar>::Slipwall(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics)
	: FlowBC<scalar>(SLIP_WALL_BC, bc_tag, gasphysics)
{ }

template <typename scalar>
void Slipwall<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                         scalar *const __restrict gs) const
{
	const a_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];
	gs[0] = ins[0];
	for(int i = 1; i < NDIM+1; i++)
		gs[i] = ins[i] - 2.0*vni*n[i-1]*ins[0];
	gs[3] = ins[3];
}

/// \todo Make this dimension-independent
template <typename scalar>
void Slipwall<scalar>::computeGhostStateAndJacobian(const scalar *const ins, const scalar *const n,
                                                    scalar *const __restrict gs,
                                                    scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const scalar vni = dimDotProduct(&ins[1],n)/ins[0];
	scalar dvni[NVARS];
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
Adiabaticwall2D<scalar>::Adiabaticwall2D(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                                         const a_real wall_tangential_velocity)
	: FlowBC<scalar>(ADIABATIC_WALL_BC, bc_tag, gasphysics), tangvel{wall_tangential_velocity}
{ }

template <typename scalar>
void Adiabaticwall2D<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                              scalar *const __restrict gs) const
{
	const a_real tangMomentum = tangvel * ins[0];
	gs[0] = ins[0];
	gs[1] =  2.0*tangMomentum*n[1] - ins[1];
	gs[2] = -2.0*tangMomentum*n[0] - ins[2];
	gs[3] = ins[3];
}

template <typename scalar>
void Adiabaticwall2D<scalar>::computeGhostStateAndJacobian(const scalar *const ins,
                                                           const scalar *const n,
                                                           scalar *const __restrict gs,
                                                           scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const a_real tangMomentum = tangvel * ins[0];
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

template <typename scalar>
Isothermalwall2D<scalar>::Isothermalwall2D(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics,
                                           const a_real wtv, const a_real temp)
	: FlowBC<scalar>(ISOTHERMAL_WALL_BC, bc_tag, gasphysics), tangvel{wtv}, walltemperature{temp}
{ }

template <typename scalar>
void Isothermalwall2D<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                              scalar *const __restrict gs) const
{
	// pressure in the interior cell
	const a_real p = phy.getPressureFromConserved(ins);
			
	// temperature in the ghost cell
	const a_real gtemp = 2.0*walltemperature - phy.getTemperature(ins[0],p);

	//gs[0] = physics.getDensityFromPressureTemperature(p, gtemp);
	gs[0] = ins[0];
	gs[1] = gs[0]*( 2.0*tangvel*n[1] - ins[1]/ins[0]);
	gs[2] = gs[0]*(-2.0*tangvel*n[0] - ins[2]/ins[0]);
	const a_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(gs[0]*gs[0]);
	gs[3] = phy.getEnergyFromTemperature(gtemp, gs[0], vmag2);
}

template <typename scalar>
void Isothermalwall2D<scalar>::computeGhostStateAndJacobian(const scalar *const ins,
                                                            const scalar *const n,
                                                            scalar *const __restrict gs,
                                                            scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	/// \todo Fix isothermal BC Jacobian
	// FIXME: Wrong Jacobian
	const a_real tangMomentum = tangvel * ins[0];
	gs[0] = ins[0];
	dgs[0] = 1.0;

	gs[1] =  2.0*tangMomentum*n[1] - ins[1];
	dgs[NVARS+0] = 2.0*tangvel*n[1];
	dgs[NVARS+1] = -1.0;

	gs[2] = -2.0*tangMomentum*n[0] - ins[2];
	dgs[2*NVARS+0] = -2.0*tangvel*n[0];
	dgs[2*NVARS+2] = -1.0;

	const a_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]);
	gs[3] = phy.getEnergyFromTemperature(walltemperature, ins[0], vmag2);

	a_real dvmag2[NVARS]; // derivative of vmag2 w.r.t. ins
	dvmag2[0] = -2.0*dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]*ins[0]);
	for(int i = 1; i < NDIM+1; i++)
		dvmag2[i] = 1.0/(ins[0]*ins[0]) * gs[i] * (-1.0);
	dvmag2[NDIM+1] = 0;

	// dummy dT for the next function call
	a_real dT[NVARS];
	for(int i = 0; i < NVARS; i++) dT[i] = 0;

	phy.getJacobianEnergyFromJacobiansTemperatureVmag2(walltemperature, ins[0], vmag2,
	                                                   dT, dvmag2, &dgs[3*NVARS]);
}

template <typename scalar>
Extrapolation<scalar>::Extrapolation(const int bc_tag, const IdealGasPhysics<scalar>& gasphysics)
	: FlowBC<scalar>(EXTRAPOLATION_BC, bc_tag, gasphysics)
{ }

template <typename scalar>
void Extrapolation<scalar>::computeGhostState(const scalar *const ins, const scalar *const n,
                                              scalar *const __restrict gs) const
{
	for(int k = 0; k < NVARS; k++) {
		gs[k] = ins[k];
	}
}

template <typename scalar>
void Extrapolation<scalar>::computeGhostStateAndJacobian(const scalar *const ins,
                                                         const scalar *const n,
                                                         scalar *const __restrict gs,
                                                         scalar *const __restrict dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;
	for(int k = 0; k < NVARS; k++) {
		gs[k] = ins[k];
		dgs[k*NVARS+k] = 1.0;
	}
}

template class InOutFlow<a_real>;
template class InFlow<a_real>;
template class Farfield<a_real>;
template class Slipwall<a_real>;
template class Adiabaticwall2D<a_real>;
template class Isothermalwall2D<a_real>;
template class Extrapolation<a_real>;

// template <typename scalar>
// std::map<int,const FlowBC<scalar>*> create_const_flowBCs(const FlowBCConfig& conf,
//                                                          const IdealGasPhysics<scalar>& physics,
//                                                          const std::array<a_real,NVARS>& uinf)
// {
// 	std::map<int,const FlowBC<scalar>*> bcmap;

// 	const FlowBC<scalar>* iobc = new InOutFlow<scalar>(conf.inflowoutflow_id, physics, uinf);
// 	bcmap[conf.inflowoutflow_id] = iobc;

// 	const FlowBC<scalar>* ibc = new InFlow<scalar>(conf.subsonicinflow_id, physics,
// 	                                               conf.subsonicinflow_ptot, conf.subsonicinflow_ttot);
// 	bcmap[conf.subsonicinflow_id] = ibc;

// 	const FlowBC<scalar>* fbc = new Farfield<scalar>(conf.farfield_id, physics, uinf);
// 	bcmap[conf.farfield_id] = fbc;

// 	const FlowBC<scalar>* ebc = new Extrapolation<scalar>(conf.extrapolation_id, physics);
// 	bcmap[conf.extrapolation_id] = ebc;

// 	const FlowBC<scalar>* slipbc = new Slipwall<scalar>(conf.slipwall_id, physics);
// 	bcmap[conf.slipwall_id] = slipbc;

// 	const FlowBC<scalar>* adiabc = new Adiabaticwall2D<scalar>(conf.adiabaticwall_id, physics,
// 	                                                           conf.adiabaticwall_vel);
// 	bcmap[conf.adiabaticwall_id] = adiabc;

// 	const FlowBC<scalar>* isotbc = new Isothermalwall2D<scalar>(conf.isothermalwall_id, physics,
// 	                                                            conf.isothermalwall_vel,
// 	                                                            conf.isothermalwall_temp);
// 	bcmap[conf.isothermalwall_id] = isotbc;

// 	return bcmap;
// }

// template
// std::map<int,const FlowBC<a_real>*> create_const_flowBCs(const FlowBCConfig& conf,
//                                                          const IdealGasPhysics<a_real>& physics,
//                                                          const std::array<a_real,NVARS>& uinf);

template <typename scalar>
std::map<int,const FlowBC<scalar>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<scalar>& physics,
                                                         const std::array<a_real,NVARS>& uinf)
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
std::map<int,const FlowBC<a_real>*> create_const_flowBCs(const std::vector<FlowBCConfig>& conf,
                                                         const IdealGasPhysics<a_real>& physics,
                                                         const std::array<a_real,NVARS>& uinf);

BCType getBCTypeFromString(const std::string bct)
{
	if(bct == "slipwall")
		return SLIP_WALL_BC;
	else if(bct == "isothermalwall")
		return ISOTHERMAL_WALL_BC;
	else if(bct == "adiabaticwall")
		return ADIABATIC_WALL_BC;
	else if(bct == "farfield")
		return FARFIELD_BC;
	else if(bct == "inflowoutflow")
		return INFLOW_OUTFLOW_BC;
	else if(bct == "extrapolation")
		return EXTRAPOLATION_BC;
	else if(bct == "subsonicinflow")
		return SUBSONIC_INFLOW_BC;
	else if(bct == "periodic")
		return PERIODIC_BC;
	else
		throw std::runtime_error("BC type not available!");
}

std::string getStringFromBCType(const BCType bct)
{
	std::string bstr;
	if(bct == SLIP_WALL_BC)
		bstr = "slipwall";
	else if(bct == ISOTHERMAL_WALL_BC)
		bstr = "isothermalwall";
	// else if(bct == "adiabaticwall")
	// 	bstr = ADIABATIC_WALL_BC;
	// else if(bct == "farfield")
	// 	bstr = FARFIELD_BC;
	// else if(bct == "inflowoutflow")
	// 	bstr = INFLOW_OUTFLOW_BC;
	// else if(bct == "extrapolation")
	// 	bstr = EXTRAPOLATION_BC;
	// else if(bct == "subsonicinflow")
	// 	bstr = SUBSONIC_INFLOW_BC;
	// else if(bct == "periodic")
	// 	bstr = PERIODIC_BC;
	else
		throw std::runtime_error("BC type not available!");

	return bstr;
}

}

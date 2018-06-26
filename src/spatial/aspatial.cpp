/** @file aspatial.cpp
 * @brief Finite volume spatial discretization of Euler/Navier-Stokes equations.
 * @author Aditya Kashi
 * @date Feb 24, 2016
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
#include <iomanip>
#include "utilities/afactory.hpp"
#include "aspatial.hpp"

namespace fvens {

/** Currently, the ghost cell coordinates are computed as reflections about the face centre.
 * \todo TODO: Replace midpoint-reflected ghost cells with face-reflected ones.
 * \sa compute_ghost_cell_coords_about_midpoint
 * \sa compute_ghost_cell_coords_about_face
 */
template<int nvars>
Spatial<nvars>::Spatial(const UMesh2dh *const mesh) : m(mesh)
{
	rc.resize(m->gnelem()+m->gnbface(), NDIM);
	gr = new amat::Array2d<a_real>[m->gnaface()];
	for(int i = 0; i <  m->gnaface(); i++)
		gr[i].resize(NGAUSS, NDIM);

	// get cell centers (real and ghost)
	
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int idim = 0; idim < NDIM; idim++)
		{
			rc(ielem,idim) = 0;
			for(int inode = 0; inode < m->gnnode(ielem); inode++)
				rc(ielem,idim) += m->gcoords(m->ginpoel(ielem, inode), idim);
			rc(ielem,idim) = rc(ielem,idim) / (a_real)(m->gnnode(ielem));
		}
	}

	a_real x1, y1, x2, y2;
	amat::Array2d<a_real> rchg(m->gnbface(),NDIM);

	compute_ghost_cell_coords_about_midpoint(rchg);
	//compute_ghost_cell_coords_about_face(rchg);

	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int relem = m->gintfac(iface,1);
		for(int idim = 0; idim < NDIM; idim++)
			rc(relem,idim) = rchg(iface,idim);
	}

	//Calculate and store coordinates of Gauss points
	// Gauss points are uniformly distributed along the face.
	for(a_int ied = 0; ied < m->gnaface(); ied++)
	{
		x1 = m->gcoords(m->gintfac(ied,2),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y2 = m->gcoords(m->gintfac(ied,3),1);
		for(int ig = 0; ig < NGAUSS; ig++)
		{
			gr[ied](ig,0) = x1 + (a_real)(ig+1.0)/(a_real)(NGAUSS+1.0) * (x2-x1);
			gr[ied](ig,1) = y1 + (a_real)(ig+1.0)/(a_real)(NGAUSS+1.0) * (y2-y1);
		}
	}
}

template<int nvars>
Spatial<nvars>::~Spatial()
{
	delete [] gr;
}

template<int nvars>
void Spatial<nvars>::compute_ghost_cell_coords_about_midpoint(amat::Array2d<a_real>& rchg)
{
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_int ip1 = m->gintfac(iface,2);
		a_int ip2 = m->gintfac(iface,3);
		a_real midpoint[NDIM];

		for(int idim = 0; idim < NDIM; idim++)
		{
			midpoint[idim] = 0.5 * (m->gcoords(ip1,idim) + m->gcoords(ip2,idim));
		}

		for(int idim = 0; idim < NDIM; idim++)
			rchg(iface,idim) = 2*midpoint[idim] - rc(ielem,idim);
	}
}

/** The ghost cell is a reflection of the boundary cell about the boundary-face.
 * It is NOT the reflection about the midpoint of the boundary-face.
 */
template<int nvars>
void Spatial<nvars>::compute_ghost_cell_coords_about_face(amat::Array2d<a_real>& rchg)
{
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_real nx = m->gfacemetric(ied,0);
		const a_real ny = m->gfacemetric(ied,1);

		const a_real xi = rc(ielem,0);
		const a_real yi = rc(ielem,1);

		const a_real x1 = m->gcoords(m->gintfac(ied,2),0);
		const a_real x2 = m->gcoords(m->gintfac(ied,3),0);
		const a_real y1 = m->gcoords(m->gintfac(ied,2),1);
		const a_real y2 = m->gcoords(m->gintfac(ied,3),1);

		// find coordinates of the point on the face that is the midpoint of the line joining
		// the real cell centre and the ghost cell centre
		a_real xs,ys;

		// check if nx != 0 and ny != 0
		if(fabs(nx)>A_SMALL_NUMBER && fabs(ny)>A_SMALL_NUMBER)		
		{
			xs = ( yi-y1 - ny/nx*xi + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
			//ys = yi + ny/nx*(xs-xi);
			ys = y1 + (y2-y1)/(x2-x1) * (xs-x1);
		}
		else if(fabs(nx)<=A_SMALL_NUMBER)
		{
			xs = xi;
			ys = y1;
		}
		else
		{
			xs = x1;
			ys = yi;
		}
		rchg(ied,0) = 2.0*xs-xi;
		rchg(ied,1) = 2.0*ys-yi;
	}
}

template <int nvars>
void Spatial<nvars>::getFaceGradient_modifiedAverage(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real gradl[NDIM][nvars], const a_real gradr[NDIM][nvars], a_real grad[NDIM][nvars])
	const
{
	a_real dr[NDIM], dist=0;
	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);

	for(int i = 0; i < NDIM; i++) {
		dr[i] = rc(relem,i)-rc(lelem,i);
		dist += dr[i]*dr[i];
	}
	dist = std::sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}

	for(int i = 0; i < nvars; i++) 
	{
		a_real davg[NDIM];
		
		for(int j = 0; j < NDIM; j++)
			davg[j] = 0.5*(gradl[j][i] + gradr[j][i]);

		const a_real corr = (ucr[i]-ucl[i])/dist;
		
		const a_real ddr = dimDotProduct(davg,dr);

		for(int j = 0; j < NDIM; j++)
		{
			grad[j][i] = davg[j] - ddr*dr[j] + corr*dr[j];
		}
	}
}

template <int nvars>
void Spatial<nvars>::getFaceGradientAndJacobian_thinLayer(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real *const dul, const a_real *const dur,
		a_real grad[NDIM][nvars], a_real dgradl[NDIM][nvars][nvars], a_real dgradr[NDIM][nvars][nvars])
	const
{
	a_real dr[NDIM], dist=0;

	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);
	for(int i = 0; i < NDIM; i++) {
		dr[i] = rc(relem,i)-rc(lelem,i);
		dist += dr[i]*dr[i];
	}
	dist = sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}

	for(int i = 0; i < nvars; i++) 
	{
		const a_real corr = (ucr[i]-ucl[i])/dist;        //< The thin layer gradient magnitude
		
		for(int j = 0; j < NDIM; j++)
		{
			grad[j][i] = corr*dr[j];
			
			for(int k = 0; k < nvars; k++) {
				dgradl[j][i][k] = -dul[i*nvars+k]/dist * dr[j];
				dgradr[j][i][k] = dur[i*nvars+k]/dist * dr[j];
			}
		}
	}
}

FlowFV_base::FlowFV_base(const UMesh2dh *const mesh,
                         const FlowPhysicsConfig& pconf, 
                         const FlowNumericsConfig& nconf)
	: 
	Spatial<NVARS>(mesh), 
	pconfig{pconf},
	nconfig{nconf},
	physics(pconfig.gamma, pconfig.Minf, pconfig.Tinf, pconfig.Reinf, pconfig.Pr), 
	uinf(physics.compute_freestream_state(pconfig.aoa)),

	inviflux {create_const_inviscidflux(nconfig.conv_numflux, &physics)}, 
	jflux {create_const_inviscidflux(nconfig.conv_numflux_jac, &physics)},

	gradcomp {create_const_gradientscheme<NVARS>(nconfig.gradientscheme, m, rc)},

	lim {create_const_reconstruction(nconfig.reconstruction, m, rc, gr, nconfig.limiter_param)}

{
	std::cout << " FlowFV: Boundary markers:\n";
	std::cout << "  Farfield " << pconfig.bcconf.farfield_id 
		<< ", inflow/outflow " << pconfig.bcconf.inflowoutflow_id
		<< ", slip wall " << pconfig.bcconf.slipwall_id;
	std::cout << "  Extrapolation " << pconfig.bcconf.extrapolation_id 
		<< ", Periodic " << pconfig.bcconf.periodic_id << '\n';
	std::cout << "  Isothermal " << pconfig.bcconf.isothermalwall_id;
	std::cout << "  Adiabatic " << pconfig.bcconf.adiabaticwall_id;
	std::cout << " FlowFV: Adiabatic wall tangential velocity = " 
		<< pconfig.bcconf.adiabaticwall_vel << '\n';
}

FlowFV_base::~FlowFV_base()
{
	delete gradcomp;
	delete inviflux;
	delete jflux;
	delete lim;
}

StatusCode FlowFV_base::initializeUnknowns(Vec u) const
{
	StatusCode ierr = 0;
	PetscScalar * uloc;
	VecGetArray(u, &uloc);
	PetscInt locsize;
	VecGetLocalSize(u, &locsize);
	assert(locsize % NVARS == 0);
	locsize /= NVARS;
	
	//initial values are equal to boundary values
	for(a_int i = 0; i < locsize; i++)
		for(int j = 0; j < NVARS; j++)
			uloc[i*NVARS+j] = uinf[j];

	VecRestoreArray(u, &uloc);

#ifdef DEBUG
	std::cout << "FlowFV: loaddata(): Initial data calculated.\n";
#endif
	return ierr;
}

void FlowFV_base::compute_boundary_states(
		const amat::Array2d<a_real>& ins, 
		       amat::Array2d<a_real>& bs ) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		compute_boundary_state(ied, &ins(ied,0), &bs(ied,0));
	
		if(m->gintfacbtags(ied,0) == pconfig.bcconf.periodic_id)
		{
			for(int i = 0; i < NVARS; i++)
				bs(ied,i) = ins(m->gperiodicmap(ied), i);
		}
	}
}

void FlowFV_base::compute_boundary_state(const int ied, 
		const a_real *const ins, 
		a_real *const gs        ) const
{
	const std::array<a_real,NDIM> n = m->gnormal(ied);

	const a_real vni = dimDotProduct(&ins[1],&n[0])/ins[0];

	if(m->gintfacbtags(ied,0) == pconfig.bcconf.slipwall_id)
	{
		gs[0] = ins[0];
		for(int i = 1; i < NDIM+1; i++)
			gs[i] = ins[i] - 2.0*vni*n[i-1]*ins[0];
		gs[3] = ins[3];
	}

	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.extrapolation_id)
	{
		gs[0] = ins[0];
		for(int i = 1; i < NDIM+1; i++)
			gs[i] = ins[i];
		gs[3] = ins[3];
	}

	/** For the far-field BCs, ghost state values are always free-stream values.
	 */
	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.farfield_id)
	{
		for(int i = 0; i < NVARS; i++)
			gs[i] = uinf[i];
	}

	/** The "inflow-outflow" BC assumes we know the state at the inlet is 
	 * the free-stream state with certainty,
	 * while the state at the outlet is not certain to be the free-stream state. 
	 * If so, we can just impose free-stream conditions for the ghost cells of inflow faces.
	 *
	 * The outflow boundary condition corresponds to Sec 2.4 "Pressure outflow boundary condition"
	 * in the paper \cite carlson_bcs. It assumes that the flow at the outflow boundary is
	 * isentropic.
	 *
	 * Whether the flow is subsonic or supersonic at the boundary
	 * is decided by interior value of the Mach number.
	 */
	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.inflowoutflow_id)
	{
		const a_real ci = physics.getSoundSpeedFromConserved(ins);
		const a_real Mni = vni/ci;
		const a_real pinf = physics.getFreestreamPressure();

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
			gs[NDIM+1] = physics.getEnergyFromPressure( pinf, ins[0],
					dimDotProduct(&ins[1],&ins[1])/(ins[0]*ins[0]) );
		}
		
		// At supersonic outflow, everything is taken from the interior
		else
		{
			for(int i = 0; i < NVARS; i++)
				gs[i] = ins[i];
		}
	}

	else if(pconfig.viscous_sim) 
	{
		if(m->gintfacbtags(ied,0) == pconfig.bcconf.adiabaticwall_id)
		{
			const a_real tangMomentum = pconfig.bcconf.adiabaticwall_vel * ins[0];
			gs[0] = ins[0];
			gs[1] =  2.0*tangMomentum*n[1] - ins[1];
			gs[2] = -2.0*tangMomentum*n[0] - ins[2];
			gs[3] = ins[3];
		}

		/** At an isothermal wall, density in the ghost cell is the same as that in the interior cell.
		 * Temperature is computed from the boundary value, and is used to compute the energy.
		 */
		else if(m->gintfacbtags(ied,0) == pconfig.bcconf.isothermalwall_id)
		{
			// pressure in the interior cell
			const a_real p = physics.getPressureFromConserved(ins);
			
			// temperature in the ghost cell
			const a_real gtemp = 2.0*pconfig.bcconf.isothermalwall_temp
				- physics.getTemperature(ins[0],p);

			//gs[0] = physics.getDensityFromPressureTemperature(p, gtemp);
			gs[0] = ins[0];
			gs[1] = gs[0]*( 2.0*pconfig.bcconf.isothermalwall_vel*n[1] - ins[1]/ins[0]);
			gs[2] = gs[0]*(-2.0*pconfig.bcconf.isothermalwall_vel*n[0] - ins[2]/ins[0]);
			const a_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(gs[0]*gs[0]);
			gs[3] = physics.getEnergyFromTemperature(gtemp, gs[0], vmag2);
		}

		else if(m->gintfacbtags(ied,0) == pconfig.bcconf.isothermalbaricwall_id)
		{
			// pressure in the ghost cell
			const a_real gp = physics.getFreestreamPressure();
			
			// temperature in the ghost cell
			const a_real gtemp= 2.0*pconfig.bcconf.isothermalbaricwall_temp 
				- physics.getTemperature(ins[0],gp);

			gs[0] = physics.getDensityFromPressureTemperature(gp, gtemp);
			gs[1] = gs[0]*( 2.0*pconfig.bcconf.isothermalbaricwall_vel*n[1] - ins[1]/ins[0]);
			gs[2] = gs[0]*(-2.0*pconfig.bcconf.isothermalbaricwall_vel*n[0] - ins[2]/ins[0]);
			const a_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(gs[0]*gs[0]);
			gs[3] = physics.getEnergyFromTemperature(gtemp, gs[0], vmag2);
		}
	}
	else if(!(m->gintfacbtags(ied,0) == pconfig.bcconf.periodic_id)) {
		// Taken care of earlier
		std::cout << " ! FlowFV: Unknown boundary tag!!\n";
		std::abort();
	}
}

void FlowFV_base::compute_boundary_Jacobian(const int ied, 
		const a_real *const ins, 
		a_real *const gs, a_real *const dgs) const
{
	for(int k = 0; k < NVARS*NVARS; k++)
		dgs[k] = 0;

	const a_real n[NDIM] = {m->gfacemetric(ied,0), m->gfacemetric(ied,1)};
	const a_real vni = dimDotProduct(&ins[1],n)/ins[0];
	const a_real dvni[NVARS] = { 
		-vni/ins[0],
		n[0]/ins[0],
		n[1]/ins[0],
		0
	};

	if(m->gintfacbtags(ied,0) == pconfig.bcconf.slipwall_id)
	{
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

	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.extrapolation_id)
	{
		gs[0] = ins[0];
		gs[1] = ins[1];
		gs[2] = ins[2];
		gs[3] = ins[3];
		for(int k = 0; k < NVARS; k++)
			dgs[k*NVARS+k] = 1.0;
	}

	/* For the far-field BCs, ghost state values are always free-stream values.
	 * Therefore the Jacobian is zero.
	 */
	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.farfield_id)
	{
		for(int i = 0; i < NVARS; i++)
			gs[i] = uinf[i];
	}

	else if(m->gintfacbtags(ied,0) == pconfig.bcconf.inflowoutflow_id)
	{
		const a_real ci = physics.getSoundSpeedFromConserved(ins);
		const a_real Mni = vni/ci;
		
		const a_real pinf = physics.getPressureFromConserved(&uinf[0]);

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
			gs[NDIM+1] = physics.getEnergyFromPressure( pinf, ins[0],
					dimDotProduct(&ins[1],&ins[1])/(ins[0]*ins[0]) );
			
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

	else if(pconfig.viscous_sim) 
	{
		if(m->gintfacbtags(ied,0) == pconfig.bcconf.adiabaticwall_id)
		{
			const a_real tangMomentum = pconfig.bcconf.adiabaticwall_vel * ins[0];
			gs[0] = ins[0];
			dgs[0] = 1.0;

			gs[1] =  2.0*tangMomentum*n[1] - ins[1];
			dgs[NVARS+0] = 2.0*pconfig.bcconf.adiabaticwall_vel*n[1];
			dgs[NVARS+1] = -1.0;

			gs[2] = -2.0*tangMomentum*n[0] - ins[2];
			dgs[2*NVARS+0] = -2.0*pconfig.bcconf.adiabaticwall_vel*n[0];
			dgs[2*NVARS+2] = -1.0;

			gs[3] = ins[3];
			dgs[3*NVARS+3] = 1.0;
		}

		else if(m->gintfacbtags(ied,0) == pconfig.bcconf.isothermalwall_id)
		{
			/// \todo Fix isothermal BC Jacobian
			// FIXME: Wrong Jacobian
			const a_real tangMomentum = pconfig.bcconf.isothermalwall_vel * ins[0];
			gs[0] = ins[0];
			dgs[0] = 1.0;

			gs[1] =  2.0*tangMomentum*n[1] - ins[1];
			dgs[NVARS+0] = 2.0*pconfig.bcconf.isothermalwall_vel*n[1];
			dgs[NVARS+1] = -1.0;

			gs[2] = -2.0*tangMomentum*n[0] - ins[2];
			dgs[2*NVARS+0] = -2.0*pconfig.bcconf.isothermalwall_vel*n[0];
			dgs[2*NVARS+2] = -1.0;

			const a_real vmag2 = dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]);
			gs[3] = physics.getEnergyFromTemperature(pconfig.bcconf.isothermalwall_temp,
			                                         ins[0], vmag2);
			
			a_real dvmag2[NVARS]; // derivative of vmag2 w.r.t. ins
			dvmag2[0] = -2.0*dimDotProduct(&gs[1],&gs[1])/(ins[0]*ins[0]*ins[0]);
			for(int i = 1; i < NDIM+1; i++)
				dvmag2[i] = 1.0/(ins[0]*ins[0]) * gs[i] * (-1.0);
			dvmag2[NDIM+1] = 0;

			// dummy dT for the next function call
			a_real dT[NVARS];
			for(int i = 0; i < NVARS; i++) dT[i] = 0;

			physics.getJacobianEnergyFromJacobiansTemperatureVmag2(pconfig.bcconf.isothermalwall_temp,
				ins[0], vmag2, dT, dvmag2, &dgs[3*NVARS]);
		}

		else if(m->gintfacbtags(ied,0) == pconfig.bcconf.isothermalbaricwall_id)
		{
			// FIXME: Wrong Jacobian
			const a_real tangMomentum = pconfig.bcconf.isothermalbaricwall_vel * ins[0];
			const a_real gp = physics.getFreestreamPressure();
			gs[0] = physics.getDensityFromPressureTemperature
				(gp, pconfig.bcconf.isothermalbaricwall_temp);
			gs[1] =  2.0*tangMomentum*n[1] - ins[1];
			gs[2] = -2.0*tangMomentum*n[0] - ins[2];
			a_real prim2state[] = {gs[0],gs[1]/gs[0],gs[2]/gs[0],
			                       pconfig.bcconf.isothermalbaricwall_temp};
			gs[3] = physics.getEnergyFromPrimitive2(prim2state);
		}
	}
	else {
		std::cout << " ! FlowFV: Cannot compute BC Jacobian - BC does not exist!!\n";
		std::abort();
	}
}

void FlowFV_base::getGradients(const MVector& u, 
                               GradVector& grads) const
{
	amat::Array2d<a_real> ug(m->gnbface(),NVARS);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface,0));
	}

	gradcomp->compute_gradients(u, ug, grads);
}

StatusCode FlowFV_base::postprocess_point(const Vec uvec, 
		amat::Array2d<a_real>& scalars, 
		amat::Array2d<a_real>& velocities) const
{
	std::cout << "FlowFV: postprocess_point(): Creating output arrays...\n";
	scalars.resize(m->gnpoin(),4);
	velocities.resize(m->gnpoin(),NDIM);
	
	amat::Array2d<a_real> areasum(m->gnpoin(),1);
	amat::Array2d<a_real> up(m->gnpoin(), NVARS);
	up.zeros();
	areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector> u(uarr, m->gnelem(), NVARS);

	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int inode = 0; inode < m->gnnode(ielem); inode++)
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u(ielem,ivar)*m->garea(ielem);
				areasum(m->ginpoel(ielem,inode)) += m->garea(ielem);
			}
	}

	for(a_int ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(int ivar = 0; ivar < NVARS; ivar++)
			up(ipoin,ivar) /= areasum(ipoin);
	
	for(a_int ipoin = 0; ipoin < m->gnpoin(); ipoin++)
	{
		scalars(ipoin,0) = up(ipoin,0);

		for(int idim = 0; idim < NDIM; idim++)
			velocities(ipoin,idim) = up(ipoin,idim+1)/up(ipoin,0);
		const a_real vmag2 = dimDotProduct(&velocities(ipoin,0),&velocities(ipoin,0));

		scalars(ipoin,2) = physics.getPressureFromConserved(&up(ipoin,0));
		a_real c = physics.getSoundSpeedFromConserved(&up(ipoin,0));
		scalars(ipoin,1) = sqrt(vmag2)/c;
		scalars(ipoin,3) = physics.getTemperatureFromConserved(&up(ipoin,0));
	}

	compute_entropy_cell(uvec);

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_point(): Done.\n";
	return ierr;
}

StatusCode FlowFV_base::postprocess_cell(const Vec uvec, 
		amat::Array2d<a_real>& scalars, 
		amat::Array2d<a_real>& velocities) const
{
	std::cout << "FlowFV: postprocess_cell(): Creating output arrays...\n";
	scalars.resize(m->gnelem(), 3);
	velocities.resize(m->gnelem(), NDIM);
	
	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector> u(uarr, m->gnelem(), NVARS);

	for(a_int iel = 0; iel < m->gnelem(); iel++) {
		scalars(iel,0) = u(iel,0);
	}

	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		velocities(iel,0) = u(iel,1)/u(iel,0);
		velocities(iel,1) = u(iel,2)/u(iel,0);
		a_real vmag2 = pow(velocities(iel,0), 2) + pow(velocities(iel,1), 2);
		scalars(iel,2) = physics.getPressureFromConserved(&uarr[iel*NVARS]);
		a_real c = physics.getSoundSpeedFromConserved(&uarr[iel*NVARS]);
		scalars(iel,1) = sqrt(vmag2)/c;
	}
	compute_entropy_cell(uvec);
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	std::cout << "FlowFV: postprocess_cell(): Done.\n";
	return ierr;
}

a_real FlowFV_base::compute_entropy_cell(const Vec uvec) const
{
	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr);

	a_real sinf = physics.getEntropyFromConserved(&uinf[0]);

	amat::Array2d<a_real> s_err(m->gnelem(),1);
	a_real error = 0;
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		s_err(iel) = (physics.getEntropyFromConserved(&uarr[iel*NVARS]) - sinf) / sinf;
		error += s_err(iel)*s_err(iel)*m->garea(iel);
	}
	error = sqrt(error);

	a_real h = 1.0/sqrt(m->gnelem());
 
	std::cout << "FlowFV:   " << log10(h) << "  " 
		<< std::setprecision(10) << log10(error) << std::endl;

	ierr = VecRestoreArrayRead(uvec, &uarr);
	(void)ierr;
	return error;
}

template<bool secondOrderRequested, bool constVisc>
FlowFV<secondOrderRequested,constVisc>::FlowFV(const UMesh2dh *const mesh,
                                               const FlowPhysicsConfig& pconf, 
                                               const FlowNumericsConfig& nconf)
	: FlowFV_base(mesh, pconf, nconf)
{
	if(secondOrderRequested)
		std::cout << "FlowFV: Second order solution requested.\n";
	if(constVisc)
		std::cout << " FLowFV: Using constant viscosity.\n";
}

template<bool secondOrderRequested, bool constVisc>
FlowFV<secondOrderRequested,constVisc>::~FlowFV()
{ }

template<bool secondOrderRequested, bool constVisc>
void FlowFV<secondOrderRequested,constVisc>::computeViscousFlux(const a_int iface, 
		const a_real *const ucell_l, const a_real *const ucell_r, const amat::Array2d<a_real>& ug,
		const std::vector<FArray<NDIM,NVARS>,aligned_allocator<FArray<NDIM,NVARS>>>& grads,
		const amat::Array2d<a_real>& ul, const amat::Array2d<a_real>& ur,
		a_real *const __restrict vflux) const
{
	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);

	/* Get proper state variables and grads at cell centres
	 * we start with all conserved variables and either conservative or primitive gradients
	 */

	// cell-centred left and right states
	a_real ucl[NVARS], ucr[NVARS];
	// left and right gradients; zero for first order scheme
	a_real gradl[NDIM][NVARS], gradr[NDIM][NVARS];
	
	for(int i = 0; i < NVARS; i++) 
	{
		ucl[i] = ucell_l[i];
		
		for(int j = 0; j < NDIM; j++) {
			gradl[j][i] = 0; 
			gradr[j][i] = 0;
		}
	}
	
	if(iface < m->gnbface())
	{
		// boundary face
		
		if(secondOrderRequested)
		{
			for(int i = 0; i < NVARS; i++) {
				ucr[i] = ug(iface,i);
			}
	
			// note: these copies are necessary because we need row-major raw C arrays for
			// the physics operations later in the function
			for(int j = 0; j < NDIM; j++)
				for(int i = 0; i < NVARS; i++) {
					gradl[j][i] = grads[lelem](j,i);
				}

			// If gradients are those of primitive variables,
			// convert cell-centred variables to primitive; we need primitive variables
			// to compute temperature gradient from primitive gradients.
			// ug was already primitive, so don't convert ucr.
			physics.getPrimitiveFromConserved(ucl, ucl);
			
			// get one-sided temperature gradients from one-sided primitive gradients
			// and discard grad p in favor of grad T.
			for(int j = 0; j < NDIM; j++)
				gradl[j][NVARS-1] = physics.getGradTemperature(ucl[0], gradl[j][0],
							ucl[NVARS-1], gradl[j][NVARS-1]);

			// Use the same gradients on both sides of a boundary face;
			// this will amount to just using the one-sided gradient for the modified average
			// gradient later.
			for(int i = 0; i < NVARS; i++) {
				gradr[0][i] = gradl[0][i]; 
				gradr[1][i] = gradl[1][i];
			}
		}
		else
		{
			// if second order was not requested, boundary values are stored in ur, not ug
			for(int i = 0; i < NVARS; i++) {
				ucr[i] = ur(iface,i);
			}
		}
	}
	else {
		for(int i = 0; i < NVARS; i++) {
			ucr[i] = ucell_r[i];
		}

		if(secondOrderRequested)
		{
			for(int j = 0; j < NDIM; j++)
				for(int i = 0; i < NVARS; i++) {
					gradl[j][i] = grads[lelem](j,i);
					gradr[j][i] = grads[relem](j,i);
				}

			physics.getPrimitiveFromConserved(ucl, ucl);
			physics.getPrimitiveFromConserved(ucr, ucr);
			
			/* get one-sided temperature gradients from one-sided primitive gradients
			 * and discard grad p in favor of grad T.
			 */
			for(int j = 0; j < NDIM; j++) {
				gradl[j][NVARS-1] = physics.getGradTemperature(ucl[0], gradl[j][0],
							ucl[NVARS-1], gradl[j][NVARS-1]);
				gradr[j][NVARS-1] = physics.getGradTemperature(ucr[0], gradr[j][0],
							ucr[NVARS-1], gradr[j][NVARS-1]);
			}
		}
	}

	// convert cell-centred variables to primitive-2
	if(secondOrderRequested)
	{
		ucl[NVARS-1] = physics.getTemperature(ucl[0], ucl[NVARS-1]);
		ucr[NVARS-1] = physics.getTemperature(ucr[0], ucr[NVARS-1]);
	}
	else
	{
		physics.getPrimitive2FromConserved(ucl, ucl);
		physics.getPrimitive2FromConserved(ucr, ucr);
	}

	/* Compute modified averages of primitive-2 variables and their gradients.
	 * This is the only finite-volume part of this function, rest is physics and chain rule.
	 */
	
	a_real grad[NDIM][NVARS];
	getFaceGradient_modifiedAverage(iface, ucl, ucr, gradl, gradr, grad);

	/* Finally, compute viscous fluxes from primitive-2 cell-centred variables, 
	 * primitive-2 face gradients and conserved face variables.
	 */
	
	// Non-dimensional dynamic viscosity divided by free-stream Reynolds number
	const a_real muRe = constVisc ? 
			physics.getConstantViscosityCoeff() 
		:
			0.5*( physics.getViscosityCoeffFromConserved(&ul(iface,0))
			+ physics.getViscosityCoeffFromConserved(&ur(iface,0)) );
	
	// Non-dimensional thermal conductivity
	const a_real kdiff = physics.getThermalConductivityFromViscosity(muRe); 

	a_real stress[NDIM][NDIM];
	for(int i = 0; i < NDIM; i++)
		for(int j = 0; j < NDIM; j++)
			stress[i][j] = 0;
	
	physics.getStressTensor(muRe, grad, stress);

	vflux[0] = 0;
	
	for(int i = 0; i < NDIM; i++)
	{
		vflux[i+1] = 0;
		for(int j = 0; j < NDIM; j++)
			vflux[i+1] -= stress[i][j] * m->gfacemetric(iface,j);
	}

	// for the energy dissipation, compute avg velocities first
	a_real vavg[NDIM];
	for(int j = 0; j < NDIM; j++)
		vavg[j] = 0.5*( ul(iface,j+1)/ul(iface,0) + ur(iface,j+1)/ur(iface,0) );

	vflux[NVARS-1] = 0;
	for(int i = 0; i < NDIM; i++)
	{
		a_real comp = 0;
		
		for(int j = 0; j < NDIM; j++)
			comp += stress[i][j]*vavg[j];       // dissipation by momentum flux (friction etc)
		
		comp += kdiff*grad[i][NVARS-1];         // dissipation by heat flux

		vflux[NVARS-1] -= comp * m->gfacemetric(iface,i);
	}

	/* vflux is assigned all negative quantities, as should be the case when the residual is
	 * assumed to be on the left of the equals sign: du/dt + r(u) = 0.
	 */
}

template<bool secondOrder, bool constVisc>
void FlowFV<secondOrder,constVisc>::computeViscousFluxJacobian(const a_int iface,
		const a_real *const ul, const a_real *const ur,
		a_real *const __restrict dvfi, a_real *const __restrict dvfj) const
{
	a_real vflux[NVARS];             // output variable to be differentiated
	a_real upr[NVARS], upl[NVARS];

	a_real dupr[NVARS*NVARS], dupl[NVARS*NVARS];
	for(int k = 0; k < NVARS*NVARS; k++) {
		dupr[k] = 0; 
		dupl[k] = 0;
	}

	physics.getPrimitive2FromConserved(ul, upl);
	physics.getPrimitive2FromConserved(ur, upr);

	physics.getJacobianPrimitive2WrtConserved(ul, dupl);
	physics.getJacobianPrimitive2WrtConserved(ur, dupr);
	
	// gradient, in each direction, of each variable
	a_real grad[NDIM][NVARS];

	/* Jacobian of the gradient in each direction of each variable at the face w.r.t. 
	 * every variable of the left state
	 */
	a_real dgradl[NDIM][NVARS][NVARS];

	/* Jacobian of the gradient in each direction of each variable at the face w.r.t. 
	 * every variable of the right state
	 */
	a_real dgradr[NDIM][NVARS][NVARS];

	getFaceGradientAndJacobian_thinLayer(iface, upl, upr, dupl, dupr, grad, dgradl, dgradr);

	/* Finally, compute viscous fluxes from primitive-2 cell-centred variables, 
	 * primitive-2 face gradients and conserved face variables.
	 */
	
	// Non-dimensional dynamic viscosity divided by free-stream Reynolds number
	const a_real muRe = constVisc ? 
			physics.getConstantViscosityCoeff() 
		:
			0.5*( physics.getViscosityCoeffFromConserved(ul)
			+ physics.getViscosityCoeffFromConserved(ur) );
	
	// Non-dimensional thermal conductivity
	const a_real kdiff = physics.getThermalConductivityFromViscosity(muRe); 

	a_real dmul[NVARS], dmur[NVARS], dkdl[NVARS], dkdr[NVARS];
	for(int k = 0; k < NVARS; k++) {
		dmul[k] = 0; dmur[k] = 0; dkdl[k] = 0; dkdr[k] = 0;
	}

	if(!constVisc) {
		physics.getJacobianSutherlandViscosityWrtConserved(ul, dmul);
		physics.getJacobianSutherlandViscosityWrtConserved(ur, dmur);
		for(int k = 0; k < NVARS; k++) {
			dmul[k] *= 0.5;
			dmur[k] *= 0.5;
		}
		physics.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmul, dkdl);
		physics.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmur, dkdr);
	}
	
	a_real stress[NDIM][NDIM], dstressl[NDIM][NDIM][NVARS], dstressr[NDIM][NDIM][NVARS];
	for(int i = 0; i < NDIM; i++)
		for(int j = 0; j < NDIM; j++) 
		{
			stress[i][j] = 0;
			for(int k = 0; k < NVARS; k++) {
				dstressl[i][j][k] = 0;
				dstressr[i][j][k] = 0;
			}
		}
	
	physics.getJacobianStress(muRe, dmul, grad, dgradl, stress, dstressl);
	physics.getJacobianStress(muRe, dmur, grad, dgradr, stress, dstressr);

	vflux[0] = 0;
	
	for(int i = 0; i < NDIM; i++)
	{
		vflux[i+1] = 0;
		for(int j = 0; j < NDIM; j++)
		{
			vflux[i+1] -= stress[i][j] * m->gfacemetric(iface,j);

			for(int k = 0; k < NVARS; k++) {
				dvfi[(i+1)*NVARS+k] += dstressl[i][j][k] * m->gfacemetric(iface,j);
				dvfj[(i+1)*NVARS+k] -= dstressr[i][j][k] * m->gfacemetric(iface,j);
			}
		}
	}

	// for the energy dissipation, compute avg velocities first
	a_real vavg[NDIM], dvavgl[NDIM][NVARS], dvavgr[NDIM][NVARS];
	for(int j = 0; j < NDIM; j++)
	{
		vavg[j] = 0.5*( ul[j+1]/ul[0] + ur[j+1]/ur[0] );

		for(int k = 0; k < NVARS; k++) {
			dvavgl[j][k] = 0;
			dvavgr[j][k] = 0;
		}
		
		dvavgl[j][0] = -0.5*ul[j+1]/(ul[0]*ul[0]);
		dvavgr[j][0] = -0.5*ur[j+1]/(ur[0]*ur[0]);
		
		dvavgl[j][j+1] = 0.5/ul[0];
		dvavgr[j][j+1] = 0.5/ur[0];
	}

	vflux[NVARS-1] = 0;
	for(int i = 0; i < NDIM; i++)
	{
		a_real comp = 0;
		a_real dcompl[NVARS], dcompr[NVARS];
		for(int k = 0; k < NVARS; k++) {
			dcompl[k] = 0;
			dcompr[k] = 0;
		}
		
		for(int j = 0; j < NDIM; j++) 
		{
			comp += stress[i][j]*vavg[j];       // dissipation by momentum flux (friction)
			
			for(int k = 0; k < NVARS; k++) {
				dcompl[k] += dstressl[i][j][k]*vavg[j] + stress[i][j]*dvavgl[j][k];
				dcompr[k] += dstressr[i][j][k]*vavg[j] + stress[i][j]*dvavgr[j][k];
			}
		}
		
		comp += kdiff*grad[i][NVARS-1];         // dissipation by heat flux

		for(int k = 0; k < NVARS; k++) {
			dcompl[k] += dkdl[k]*grad[i][NVARS-1] + kdiff*dgradl[i][NVARS-1][k];
			dcompr[k] += dkdr[k]*grad[i][NVARS-1] + kdiff*dgradr[i][NVARS-1][k];
		}

		vflux[NVARS-1] -= comp * m->gfacemetric(iface,i);

		for(int k = 0; k < NVARS; k++) {
			dvfi[(NVARS-1)*NVARS+k] += dcompl[k] * m->gfacemetric(iface,i);
			dvfj[(NVARS-1)*NVARS+k] -= dcompr[k] * m->gfacemetric(iface,i);
		}
	}
}

template<bool secondOrder, bool constVisc>
void FlowFV<secondOrder,constVisc>::computeViscousFluxApproximateJacobian(const a_int iface,
		const a_real *const ul, const a_real *const ur,
		a_real *const __restrict dvfi, a_real *const __restrict dvfj) const
{
	// compute non-dimensional viscosity and thermal conductivity
	const a_real muRe = constVisc ? 
			physics.getConstantViscosityCoeff() 
		:
			0.5*( physics.getViscosityCoeffFromConserved(ul)
			+ physics.getViscosityCoeffFromConserved(ur) );
	
	const a_real rho = 0.5*(ul[0]+ur[0]);

	// the vector from the left cell-centre to the right, and its magnitude
	a_real dr[NDIM], dist=0;

	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);
	for(int i = 0; i < NDIM; i++) {
		dr[i] = rc(relem,i)-rc(lelem,i);
		dist += dr[i]*dr[i];
	}
	
	dist = sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}
	
	for(int i = 0; i < NVARS; i++)
	{
		dvfi[i*NVARS+i] -= muRe/(rho*dist);
		dvfj[i*NVARS+i] -= muRe/(rho*dist);
	}
}

template<bool secondOrderRequested, bool constVisc>
StatusCode FlowFV<secondOrderRequested,constVisc>::compute_residual(const Vec uvec, 
		Vec __restrict rvec, 
		const bool gettimesteps, std::vector<a_real>& dtm) const
{
	StatusCode ierr = 0;
	amat::Array2d<a_real> integ, ug, uleft, uright;	
	integ.resize(m->gnelem(), 1);
	ug.resize(m->gnbface(),NVARS);
	uleft.resize(m->gnaface(), NVARS);
	uright.resize(m->gnaface(), NVARS);
	std::vector<FArray<NDIM,NVARS>, aligned_allocator<FArray<NDIM,NVARS>> > grads;

	PetscInt locnelem; const PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());
	//ierr = VecGetLocalSize(dtmvec, &dtsz); CHKERRQ(ierr);
	//assert(locnelem == dtsz);

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector> u(uarr, locnelem, NVARS);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	Eigen::Map<MVector> residual(rarr, locnelem, NVARS);
	//ierr = VecGetArray(dtmvec, &dtm); CHKERRQ(ierr);

#pragma omp parallel default(shared)
	{
#pragma omp for simd
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			integ(iel) = 0.0;
		}

		// first, set cell-centered values of boundary cells as left-side values of boundary faces
#pragma omp for
		for(a_int ied = 0; ied < m->gnbface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			for(int ivar = 0; ivar < NVARS; ivar++)
				uleft(ied,ivar) = u(ielem,ivar);
		}
	}

	if(secondOrderRequested)
	{
		// for storing cell-centred gradients at interior cells and ghost cells
		/*dudx.resize(m->gnelem()+m->gnbface(), NVARS);
		dudy.resize(m->gnelem()+m->gnbface(), NVARS);*/
		grads.resize(m->gnelem());

		// get cell average values at ghost cells using BCs
		compute_boundary_states(uleft, ug);

		MVector up(m->gnelem(), NVARS);

		// convert everything to primitive variables
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iface = 0; iface < m->gnbface(); iface++)
			{
				physics.getPrimitiveFromConserved(&ug(iface,0), &ug(iface,0));
			}

#pragma omp for
			for(a_int iel = 0; iel < m->gnelem(); iel++)
				physics.getPrimitiveFromConserved(&uarr[iel*NVARS], &up(iel,0));
		}

		// reconstruct
		gradcomp->compute_gradients(up, ug, grads);
		lim->compute_face_values(up, ug, grads, uleft, uright);

		// Convert face values back to conserved variables - gradients stay primitive.
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
			{
				physics.getConservedFromPrimitive(&uleft(iface,0), &uleft(iface,0));
				physics.getConservedFromPrimitive(&uright(iface,0), &uright(iface,0));
			}
#pragma omp for
			for(a_int iface = 0; iface < m->gnbface(); iface++) 
			{
				physics.getConservedFromPrimitive(&uleft(iface,0), &uleft(iface,0));
			}
		}
	}
	else
	{
		// if order is 1, set the face data same as cell-centred data for all faces
		
		// set both left and right states for all interior faces
#pragma omp parallel for default(shared)
		for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
		{
			a_int ielem = m->gintfac(ied,0);
			a_int jelem = m->gintfac(ied,1);
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				uleft(ied,ivar) = u(ielem,ivar);
				uright(ied,ivar) = u(jelem,ivar);
			}
		}
	}

	// set right (ghost) state for boundary faces
	compute_boundary_states(uleft,uright);

	/** Compute fluxes.
	 * The integral of the maximum magnitude of eigenvalue over each face is also computed:
	 * \f[
	 * \int_{f_i} (|v_n| + c) \mathrm{d}l
	 * \f]
	 * so that time steps can be calculated for explicit time stepping.
	 */

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ied = 0; ied < m->gnaface(); ied++)
		{
			a_real n[NDIM];
			n[0] = m->gfacemetric(ied,0);
			n[1] = m->gfacemetric(ied,1);
			a_real len = m->gfacemetric(ied,2);
			const int lelem = m->gintfac(ied,0);
			const int relem = m->gintfac(ied,1);
			a_real fluxes[NVARS];

			inviflux->get_flux(&uleft(ied,0), &uright(ied,0), n, fluxes);

			// integrate over the face
			for(int ivar = 0; ivar < NVARS; ivar++)
					fluxes[ivar] *= len;

			if(pconfig.viscous_sim) 
			{
				// get viscous fluxes
				a_real vflux[NVARS];
				const a_real *const urt = (ied < m->gnbface()) ? nullptr : &uarr[relem*NVARS];
				computeViscousFlux(ied, &uarr[lelem*NVARS], urt, ug, grads, uleft, uright, 
						vflux);

				for(int ivar = 0; ivar < NVARS; ivar++)
					fluxes[ivar] += vflux[ivar]*len;
			}

			/// We assemble the negative of the residual ( M du/dt + r(u) = 0).
			for(int ivar = 0; ivar < NVARS; ivar++) {
#pragma omp atomic
				residual(lelem,ivar) -= fluxes[ivar];
			}
			if(relem < m->gnelem()) {
				for(int ivar = 0; ivar < NVARS; ivar++) {
#pragma omp atomic
					residual(relem,ivar) += fluxes[ivar];
				}
			}
			
			// compute max allowable time steps
			if(gettimesteps) 
			{
				//calculate speeds of sound
				const a_real ci = physics.getSoundSpeedFromConserved(&uleft(ied,0));
				const a_real cj = physics.getSoundSpeedFromConserved(&uright(ied,0));
				//calculate normal velocities
				const a_real vni = (uleft(ied,1)*n[0] +uleft(ied,2)*n[1])/uleft(ied,0);
				const a_real vnj = (uright(ied,1)*n[0] + uright(ied,2)*n[1])/uright(ied,0);

				a_real specradi = (fabs(vni)+ci)*len, specradj = (fabs(vnj)+cj)*len;

				if(pconfig.viscous_sim) 
				{
					a_real mui, muj;
					if(constVisc) {
						mui = physics.getConstantViscosityCoeff();
						muj = physics.getConstantViscosityCoeff();
					}
					else {
						mui = physics.getViscosityCoeffFromConserved(&uleft(ied,0));
						muj = physics.getViscosityCoeffFromConserved(&uright(ied,0));
					}
					a_real coi = std::max(4.0/(3*uleft(ied,0)), physics.g/uleft(ied,0));
					a_real coj = std::max(4.0/(3*uright(ied,0)), physics.g/uright(ied,0));
					
					specradi += coi*mui/physics.Pr * len*len/m->garea(lelem);
					if(relem < m->gnelem())
						specradj += coj*muj/physics.Pr * len*len/m->garea(relem);
				}

#pragma omp atomic
				integ(lelem) += specradi;
				
				if(relem < m->gnelem()) {
#pragma omp atomic
					integ(relem) += specradj;
				}
			}
		}

#pragma omp barrier

		if(gettimesteps)
#pragma omp for simd
			for(a_int iel = 0; iel < m->gnelem(); iel++)
			{
				dtm[iel] = m->garea(iel)/integ(iel);
			}
	} // end parallel region
	
	VecRestoreArrayRead(uvec, &uarr);
	VecRestoreArray(rvec, &rarr);
	//VecRestoreArray(dtmvec, &dtm);
	return ierr;
}

template<bool order2, bool constVisc>
StatusCode FlowFV<order2,constVisc>::compute_jacobian(const Vec uvec, Mat A) const
{
	StatusCode ierr = 0;

	PetscInt locnelem; const PetscScalar *uarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	//Eigen::Map<const MVector> u(uarr, m->gnelem(), NVARS);

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		a_real n[NDIM];
		n[0] = m->gfacemetric(iface,0);
		n[1] = m->gfacemetric(iface,1);
		const a_real len = m->gfacemetric(iface,2);
		
		a_real uface[NVARS];
		Matrix<a_real,NVARS,NVARS,RowMajor> drdl;
		Matrix<a_real,NVARS,NVARS,RowMajor> left;
		Matrix<a_real,NVARS,NVARS,RowMajor> right;
		
		compute_boundary_Jacobian(iface, &uarr[lelem*NVARS], uface, &drdl(0,0));	
		
		jflux->get_jacobian(&uarr[lelem*NVARS], uface, n, &left(0,0), &right(0,0));

		if(pconfig.viscous_sim) {
			//computeViscousFluxApproximateJacobian(iface, &uarr[lelem*NVARS], uface, 
			//		&left(0,0), &right(0,0));
			computeViscousFluxJacobian(iface,&uarr[lelem*NVARS],uface, &left(0,0), &right(0,0));
		}
		
		/* The actual derivative is  dF/dl  +  dF/dr * dr/dl.
		 * We actually need to subtract dF/dr from dF/dl because the inviscid numerical flux
		 * computation returns the negative of dF/dl but positive dF/dr. The latter was done to
		 * get correct signs for lower and upper off-diagonal blocks.
		 *
		 * Integrate the results over the face and negate, as -ve of L is added to D
		 */
		left = -len*(left - right*drdl);

#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1,&lelem, 1,&lelem, left.data(), ADD_VALUES);
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		//const a_int intface = iface-m->gnbface();
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);
		a_real n[NDIM];
		n[0] = m->gfacemetric(iface,0);
		n[1] = m->gfacemetric(iface,1);
		const a_real len = m->gfacemetric(iface,2);
		Matrix<a_real,NVARS,NVARS,RowMajor> L;
		Matrix<a_real,NVARS,NVARS,RowMajor> U;
	
		// NOTE: the values of L and U get REPLACED here, not added to
		jflux->get_jacobian(&uarr[lelem*NVARS], &uarr[relem*NVARS], n, &L(0,0), &U(0,0));

		if(pconfig.viscous_sim) {
			//computeViscousFluxApproximateJacobian(iface, &uarr[lelem*NVARS], &uarr[relem*NVARS], 
			//		&L(0,0), &U(0,0));
			computeViscousFluxJacobian(iface, &uarr[lelem*NVARS], &uarr[relem*NVARS], 
					&L(0,0), &U(0,0));
		}

		L *= len; U *= len;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &relem, U.data(), ADD_VALUES);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0; U *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &relem, U.data(), ADD_VALUES);
		}
	}

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	
	return ierr;
}

#if 0

/** Computes the Jacobian in a block diagonal, lower and upper format.
 * If the (numerical) flux from cell i to cell j is \f$ F_{ij}(u_i, u_j, n_{ij}) \f$,
 * then \f$ L_{ij} = -\frac{\partial F_{ij}}{\partial u_i} \f$ and
 * \f$ U_{ij} = \frac{\partial F_{ij}}{\partial u_j} \f$.
 * Also, the contribution of face ij to diagonal blocks are 
 * \f$ D_{ii} \rightarrow D_{ii} -L_{ij}, D_{jj} \rightarrow D_{jj} -U_{ij} \f$.
 */
template<bool order2, bool constVisc>
void FlowFV<order2,constVisc>::compute_jacobian(const MVector& u, 
				AbstractMatrix<a_real,a_int> *const __restrict A) const
{
#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		a_real n[NDIM];
		n[0] = m->gfacemetric(iface,0);
		n[1] = m->gfacemetric(iface,1);
		a_real len = m->gfacemetric(iface,2);
		
		a_real uface[NVARS];
		Matrix<a_real,NVARS,NVARS,RowMajor> drdl;
		Matrix<a_real,NVARS,NVARS,RowMajor> left;
		Matrix<a_real,NVARS,NVARS,RowMajor> right;
		
		compute_boundary_Jacobian(iface, &u(lelem,0), uface, &drdl(0,0));	
		
		jflux->get_jacobian(&u(lelem,0), uface, n, &left(0,0), &right(0,0));

		if(pconfig.viscous_sim) {
			computeViscousFluxApproximateJacobian(iface,&u(lelem,0),uface, &left(0,0), &right(0,0));
			//computeViscousFluxJacobian(iface,&u(lelem,0),uface, &left(0,0), &right(0,0));
		}
		
		/* The actual derivative is  dF/dl  +  dF/dr * dr/dl.
		 * We actually need to subtract dF/dr from dF/dl because the inviscid numerical flux
		 * computation returns the negative of dF/dl but positive dF/dr. The latter was done to
		 * get correct signs for lower and upper off-diagonal blocks.
		 *
		 * Integrate the results over the face and negate, as -ve of L is added to D
		 */
		left = -len*(left - right*drdl);
	
		A->updateDiagBlock(lelem*NVARS, left.data(), NVARS);
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int intface = iface-m->gnbface();
		a_int lelem = m->gintfac(iface,0);
		a_int relem = m->gintfac(iface,1);
		a_real n[NDIM];
		n[0] = m->gfacemetric(iface,0);
		n[1] = m->gfacemetric(iface,1);
		a_real len = m->gfacemetric(iface,2);
		Matrix<a_real,NVARS,NVARS,RowMajor> L;
		Matrix<a_real,NVARS,NVARS,RowMajor> U;
	
		/// NOTE: the values of L and U get REPLACED here, not added to
		jflux->get_jacobian(&u(lelem,0), &u(relem,0), n, &L(0,0), &U(0,0));

		if(pconfig.viscous_sim) {
			computeViscousFluxApproximateJacobian(iface, &u(lelem,0), &u(relem,0), 
					&L(0,0), &U(0,0));
			//computeViscousFluxJacobian(iface, &u(lelem,0), &u(relem,0), &L(0,0), &U(0,0));
		}

		L *= len; U *= len;
		if(A->type()=='d') {
			A->submitBlock(relem*NVARS,lelem*NVARS, L.data(), 1,intface);
			A->submitBlock(lelem*NVARS,relem*NVARS, U.data(), 2,intface);
		}
		else {
			A->submitBlock(relem*NVARS,lelem*NVARS, L.data(), NVARS,NVARS);
			A->submitBlock(lelem*NVARS,relem*NVARS, U.data(), NVARS,NVARS);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0; U *= -1.0;
		A->updateDiagBlock(lelem*NVARS, L.data(), NVARS);
		A->updateDiagBlock(relem*NVARS, U.data(), NVARS);
	}
}

#endif

template class FlowFV<true,true>;
template class FlowFV<false,true>;
template class FlowFV<true,false>;
template class FlowFV<false,false>;


template<int nvars>
Diffusion<nvars>::Diffusion(const UMesh2dh *const mesh, const a_real diffcoeff, const a_real bvalue,
		std::function< 
		void(const a_real *const, const a_real, const a_real *const, a_real *const)
			> sourcefunc)
	: Spatial<nvars>(mesh), diffusivity{diffcoeff}, bval{bvalue}, source(sourcefunc)
{
	h.resize(m->gnelem());
	for(a_int iel = 0; iel < m->gnelem(); iel++) {
		h[iel] = 0;
		// max face length
		for(int ifael = 0; ifael < m->gnfael(iel); ifael++) {
			a_int face = m->gelemface(iel,ifael);
			if(h[iel] < m->gfacemetric(face,2)) h[iel] = m->gfacemetric(face,2);
		}
	}
}

template<int nvars>
Diffusion<nvars>::~Diffusion()
{ }

template<int nvars>
StatusCode Diffusion<nvars>::initializeUnknowns(Vec u) const
{
	PetscScalar *uarr;
	StatusCode ierr = VecGetArray(u, &uarr); CHKERRQ(ierr);
	for(a_int i = 0; i < m->gnelem(); i++)
		for(a_int j = 0; j < nvars; j++)
			uarr[i*nvars+j] = 0;
	ierr = VecRestoreArray(u, &uarr); CHKERRQ(ierr);
	return ierr;
}

// Currently, all boundaries are constant Dirichlet
template<int nvars>
inline void Diffusion<nvars>::compute_boundary_state(const int ied, 
		const a_real *const ins, a_real *const bs) const
{
	for(int ivar = 0; ivar < nvars; ivar++)
		bs[ivar] = 2.0*bval - ins[ivar];
}

template<int nvars>
void Diffusion<nvars>::compute_boundary_states(const amat::Array2d<a_real>& instates, 
                                                amat::Array2d<a_real>& bounstates) const
{
	for(a_int ied = 0; ied < m->gnbface(); ied++)
		compute_boundary_state(ied, &instates(ied,0), &bounstates(ied,0));
}

template<int nvars>
StatusCode Diffusion<nvars>::postprocess_point(const Vec uvec, amat::Array2d<a_real>& up,
		amat::Array2d<a_real>& vec) const
{
	std::cout << "Diffusion: postprocess_point(): Creating output arrays\n";
	
	std::vector<a_real> areasum(m->gnpoin(),0);
	up.resize(m->gnpoin(), nvars);
	up.zeros();
	//areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector> u(uarr, m->gnelem(), NVARS);

	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int inode = 0; inode < m->gnnode(ielem); inode++)
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u(ielem,ivar)*m->garea(ielem);
				areasum[m->ginpoel(ielem,inode)] += m->garea(ielem);
			}
	}

	for(a_int ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(int ivar = 0; ivar < nvars; ivar++)
			up(ipoin,ivar) /= areasum[ipoin];

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	return ierr;
}

template<int nvars>
DiffusionMA<nvars>::DiffusionMA(const UMesh2dh *const mesh, 
		const a_real diffcoeff, const a_real bvalue,
	std::function<void(const a_real *const,const a_real,const a_real *const,a_real *const)> sf, 
		const std::string grad_scheme)
	: Diffusion<nvars>(mesh, diffcoeff, bvalue, sf),
	  gradcomp {create_const_gradientscheme<nvars>(grad_scheme, m, rc)}
{ }

template<int nvars>
DiffusionMA<nvars>::~DiffusionMA()
{
	delete gradcomp;
}

template<int nvars>
StatusCode DiffusionMA<nvars>::compute_residual(const Vec uvec,
                                          Vec rvec, 
                                          const bool gettimesteps, 
										  std::vector<a_real>& dtm) const
{
	StatusCode ierr = 0;

	PetscInt locnelem; const PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector> u(uarr, locnelem, nvars);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);
	Eigen::Map<MVector> residual(rarr, locnelem, nvars);

	amat::Array2d<a_real> uleft;
	amat::Array2d<a_real> ug;
	uleft.resize(m->gnbface(),nvars);	// Modified
	ug.resize(m->gnbface(),nvars);

	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		for(int ivar = 0; ivar < nvars; ivar++)
			uleft(ied,ivar) = u(ielem,ivar);
	}

	std::vector<FArray<NDIM,nvars>,aligned_allocator<FArray<NDIM,nvars>>> grads;
	grads.resize(m->gnelem());
	
	compute_boundary_states(uleft, ug);
	gradcomp->compute_gradients(u, ug, grads);

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);
		const a_real len = m->gfacemetric(iface,2);
		
		a_real gradl[NDIM][nvars], gradr[NDIM][nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim][ivar] = grads[lelem](idim,ivar);
				gradr[idim][ivar] = grads[relem](idim,ivar);
			}
		}
	
		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage(iface, &uarr[lelem*nvars], &uarr[relem*nvars], gradl, gradr, 
				gradf);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(-grad u . n) * l
			a_real flux = 0;
			for(int idim = 0; idim < NDIM; idim++)
				flux += gradf[idim][ivar]*m->gfacemetric(iface,idim);
			flux *= (-diffusivity*len);

			/// NOTE: we assemble the negative of the residual r in 'M du/dt + r(u) = 0'
#pragma omp atomic
			residual(lelem,ivar) -= flux;
#pragma omp atomic
			residual(relem,ivar) += flux;
		}
	}
	
#pragma omp parallel for default(shared)
	for(int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_real len = m->gfacemetric(iface,2);
		
		a_real gradl[NDIM][nvars], gradr[NDIM][nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim][ivar] = grads[lelem](idim,ivar);
				gradr[idim][ivar] = grads[lelem](idim,ivar);
			}
		}

		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage(iface, &uarr[lelem*nvars], &ug(iface,0), gradl, gradr, gradf);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(-grad u . n) * l
			a_real flux = 0;
			for(int idim = 0; idim < NDIM; idim++)
				flux += gradf[idim][ivar]*m->gfacemetric(iface,idim);
			flux *= (-diffusivity*len);

			/// NOTE: we assemble the negative of the residual r in 'M du/dt + r(u) = 0'
#pragma omp atomic
			residual(lelem,ivar) -= flux;
		}
	}

#pragma omp parallel for default(shared)
	for(int iel = 0; iel < m->gnelem(); iel++) 
	{
		if(gettimesteps)
			dtm[iel] = h[iel]*h[iel]/diffusivity;

		// subtract source term
		a_real sourceterm[nvars];
		source(&rc(iel,0), 0, &uarr[iel*nvars], sourceterm);
		for(int ivar = 0; ivar < nvars; ivar++)
			residual(iel,ivar) += sourceterm[ivar]*m->garea(iel);
	}
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(rvec, &rarr); CHKERRQ(ierr);
	return ierr;
}

/** For now, this is the same as the thin-layer Jacobian
 */
template<int nvars>
StatusCode DiffusionMA<nvars>::compute_jacobian(const Vec uvec,
		Mat A) const
{
	StatusCode ierr = 0;

	PetscInt locnelem; const PetscScalar *uarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		//a_int intface = iface-m->gnbface();
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);
		const a_real len = m->gfacemetric(iface,2);

		a_real du[nvars*nvars];
		for(int i = 0; i < nvars; i++) {
			for(int j = 0; j < nvars; j++)
				du[i*nvars+j] = 0;
			du[i*nvars+i] = 1.0;
		}

		a_real grad[NDIM][nvars], dgradl[NDIM][nvars][nvars], dgradr[NDIM][nvars][nvars];

		// Compute the face gradient Jacobian; we don't actually need the gradient, however..
		getFaceGradientAndJacobian_thinLayer(iface, &uarr[lelem*nvars], &uarr[relem*nvars],
				du, du, grad, dgradl, dgradr);

		a_real dfluxl[nvars*nvars];
		zeros(dfluxl, nvars*nvars);	
		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(d(-grad u)/du_l . n) * l
			for(int idim = 0; idim < NDIM; idim++)
				dfluxl[ivar*nvars+ivar] += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
			dfluxl[ivar*nvars+ivar] *= (-diffusivity*len);
		}

#pragma omp critical
		ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &lelem, dfluxl, ADD_VALUES);
#pragma omp critical
		ierr = MatSetValuesBlocked(A, 1, &relem, 1, &relem, dfluxl, ADD_VALUES);
		
		for(int ivar = 0; ivar < nvars; ivar++)
			dfluxl[ivar*nvars+ivar] *= -1;

#pragma omp critical
		ierr = MatSetValuesBlocked(A, 1, &relem, 1, &lelem, dfluxl, ADD_VALUES);
#pragma omp critical
		ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &relem, dfluxl, ADD_VALUES);
	}
	
#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_real len = m->gfacemetric(iface,2);
		
		a_real du[nvars*nvars];
		for(int i = 0; i < nvars; i++) {
			for(int j = 0; j < nvars; j++)
				du[i*nvars+j] = 0;
			du[i*nvars+i] = 1.0;
		}

		a_real grad[NDIM][nvars], dgradl[NDIM][nvars][nvars], dgradr[NDIM][nvars][nvars];

		// Compute the face gradient and its Jacobian; we don't actually need the gradient, however
		getFaceGradientAndJacobian_thinLayer(iface, &uarr[lelem*nvars], &uarr[lelem*nvars],
				du, du, grad, dgradl, dgradr);

		a_real dfluxl[nvars*nvars];
		zeros(dfluxl, nvars*nvars);	
		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(d(-grad u)/du_l . n) * l
			for(int idim = 0; idim < NDIM; idim++)
				dfluxl[ivar*nvars+ivar] += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
			dfluxl[ivar*nvars+ivar] *= (-diffusivity*len);
		}
		
#pragma omp critical
		ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &lelem, dfluxl, ADD_VALUES);
	}
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	return ierr;
}

template <int nvars>
void DiffusionMA<nvars>::getGradients(const MVector& u,
		std::vector<FArray<NDIM,nvars>,aligned_allocator<FArray<NDIM,nvars>>>& grads) const
{
	amat::Array2d<a_real> ug(m->gnbface(),nvars);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface,0));
	}

	gradcomp->compute_gradients(u, ug, grads);
}

// template instantiations

template class Diffusion<1>;
template class DiffusionMA<1>;

}	// end namespace

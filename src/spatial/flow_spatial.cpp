/** /file
 * /brief Implementation if spatial discretization for Euler/Navier-Stokes equations.
 * /author Aditya Kashi
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
#include "physics/viscousphysics.hpp"
#include "utilities/afactory.hpp"
#include "abctypemap.hpp"
#include "flow_spatial.hpp"

namespace fvens {

template <typename scalar>
FlowFV_base<scalar>::FlowFV_base(const UMesh2dh<scalar> *const mesh,
                                 const FlowPhysicsConfig& pconf, 
                                 const FlowNumericsConfig& nconf)
	: 
	Spatial<scalar,NVARS>(mesh), 
	pconfig{pconf},
	nconfig{nconf},
	physics(pconfig.gamma, pconfig.Minf, pconfig.Tinf, pconfig.Reinf, pconfig.Pr), 
	uinf(physics.compute_freestream_state(pconfig.aoa)),

	inviflux {create_const_inviscidflux<scalar>(nconfig.conv_numflux, &physics)}, 

	gradcomp {create_const_gradientscheme<scalar,NVARS>(nconfig.gradientscheme, m, rc)},
	lim {create_const_reconstruction<scalar,NVARS>(nconfig.reconstruction, m, rc, gr,
	                                               nconfig.limiter_param)},

	bcs {create_const_flowBCs<scalar>(pconf.bcconf, physics,uinf)}

{
	std::cout << " FlowFV_base: Boundary conditions:\n";
	for(auto it = pconfig.bcconf.begin(); it != pconfig.bcconf.end(); it++) {
		std::cout << "  " << bcTypeMap.left.find(it->bc_type)->second << '\n';
	}
}

template <typename scalar>
FlowFV_base<scalar>::~FlowFV_base()
{
	delete gradcomp;
	delete inviflux;
	delete lim;
	// delete BCs
	for(auto it = bcs.begin(); it != bcs.end(); it++) {
		delete it->second;
	}
}

template <typename scalar>
StatusCode FlowFV_base<scalar>::initializeUnknowns(Vec u) const
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

template <typename scalar>
void FlowFV_base<scalar>::compute_boundary_states(const amat::Array2d<scalar>& ins, 
                                                  amat::Array2d<scalar>& bs ) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		compute_boundary_state(ied, &ins(ied,0), &bs(ied,0));
	
		// if(m->gintfacbtags(ied,0) == pconfig.bcconf.periodic_id)
		// {
		// 	for(int i = 0; i < NVARS; i++)
		// 		bs(ied,i) = ins(m->gperiodicmap(ied), i);
		// }
	}
}

template <typename scalar>
void FlowFV_base<scalar>::compute_boundary_state(const int ied, 
                                         const scalar *const ins, 
                                         scalar *const gs        ) const
{
	const std::array<scalar,NDIM> n = m->gnormal(ied);
	bcs.at(m->gintfacbtags(ied,0))->computeGhostState(ins, &n[0], gs);
}

template <typename scalar>
void FlowFV_base<scalar>::getGradients(const MVector<scalar>& u,
                               GradArray<scalar,NVARS>& grads) const
{
	amat::Array2d<scalar> ug(m->gnbface(),NVARS);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface,0));
	}

	gradcomp->compute_gradients(u, ug, grads);
}

template <typename scalar>
StatusCode FlowFV_base<scalar>::assemble_residual(const Vec uvec, 
                                                  Vec __restrict rvec, 
                                                  const bool gettimesteps,
                                                  std::vector<a_real>& dtm) const
{
	StatusCode ierr = 0;
	amat::Array2d<a_real> integ, ug, uleft, uright;	
	integ.resize(m->gnelem(), 1);
	ug.resize(m->gnbface(),NVARS);
	uleft.resize(m->gnaface(), NVARS);
	uright.resize(m->gnaface(), NVARS);
	GradArray<a_real,NVARS> grads;

	PetscInt locnelem; const PetscScalar *uarr; PetscScalar *rarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());
	//ierr = VecGetLocalSize(dtmvec, &dtsz); CHKERRQ(ierr);
	//assert(locnelem == dtsz);

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);

	compute_residual(uarr, rarr, gettimesteps, dtm);
	
	VecRestoreArrayRead(uvec, &uarr);
	VecRestoreArray(rvec, &rarr);
	//VecRestoreArray(dtmvec, &dtm);
	return ierr;
}

template <typename scalar>
static inline std::array<scalar,NDIM> flowDirectionVector(const scalar aoa) {
	std::array<scalar,NDIM> dir;
	for(int i = 0; i < NDIM; i++) dir[i] = 0;

	dir[0] = cos(aoa);
	dir[1] = sin(aoa);

	return dir;
}

template <typename scalar>
std::tuple<scalar,scalar,scalar>
FlowFV_base<scalar>::computeSurfaceData (const MVector<scalar>& u,
                                         const GradArray<scalar,NVARS>& grad,
                                         const int iwbcm,
                                         MVector<scalar>& output) const
{
	// unit vector in the direction of flow
	const std::array<scalar,NDIM> av = flowDirectionVector(pconfig.aoa); 

	a_int facecoun = 0;			// face iteration counter for this boundary marker
	scalar totallen = 0;		// total area of the surface with this boundary marker
	scalar Cdf=0, Cdp=0, Cl=0;
	
	const scalar pinf = physics.getFreestreamPressure();

	// unit vector normal to the free-stream flow direction
	// TODO: Generalize to 3D
	scalar flownormal[NDIM]; flownormal[0] = -av[1]; flownormal[1] = av[0];

	// iterate over faces having this boundary marker
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		if(m->gintfacbtags(iface,0) == iwbcm)
		{
			const a_int lelem = m->gintfac(iface,0);
			scalar n[NDIM];
			for(int j = 0; j < NDIM; j++)
				n[j] = m->gfacemetric(iface,j);
			const scalar len = m->gfacemetric(iface,2);
			totallen += len;

			// coords of face center
			a_int ijp[NDIM];
			ijp[0] = m->gintfac(iface,2);
			ijp[1] = m->gintfac(iface,3);
			scalar coord[NDIM];
			for(int j = 0; j < NDIM; j++) 
			{
				coord[j] = 0;
				for(int inofa = 0; inofa < m->gnnofa(); inofa++)
					coord[j] += m->gcoords(ijp[inofa],j);
				coord[j] /= m->gnnofa();
				
				output(facecoun,j) = coord[j];
			}

			/** Pressure coefficient: 
			 * \f$ C_p = (p-p_\infty)/(\frac12 rho_\infty * v_\infty^2) \f$
			 * = 2(p* - p_inf*) where *'s indicate non-dimensional values.
			 * We note that p_inf* = 1/(gamma Minf^2) in our non-dimensionalization.
			 */
			output(facecoun, NDIM) = (physics.getPressureFromConserved(&u(lelem,0)) - pinf)*2.0;

			/** Skin friction coefficient \f% C_f = \tau_w / (\frac12 \rho v_\infty^2) \f$.
			 * 
			 * We can define \f$ \tau_w \f$, the wall shear stress, as
			 * \f$ \tau_w = (\mathbf{T} \hat{\mathbf{n}}).\hat{\mathbf{t}} \f$
			 * where \f$ \mathbf{\Tau} \f$ is the viscous stress tensor, 
			 * \f$ \hat{\mathbf{n}} \f$ is the unit normal to the face and 
			 * \f$ \hat{\mathbf{t}} \f$ is a consistent unit tangent to the face.
			 * 
			 * Note that because of our non-dimensionalization,
			 * \f$ C_f = 2 \tau_w \f$.
			 *
			 * Note that finally the wall shear stress becomes
			 * \f$ \tau_w = \mu (\nabla\mathbf{u}+\nabla\mathbf{u}^T) \hat{\mathbf{n}}
			 *                                           .\hat{\mathbf{t}} \f$.
			 *
			 * Note that if n is (n1,n2), t is chosen as (n2,-n1).
			 */

			// non-dim viscosity / Re_inf
			const scalar muhat = physics.getViscosityCoeffFromConserved(&u(lelem,0));

			// velocity gradient tensor
			scalar gradu[NDIM][NDIM];
			gradu[0][0] = (grad[lelem](0,1)*u(lelem,0)-u(lelem,1)*grad[lelem](0,0))
							/ (u(lelem,0)*u(lelem,0));
			gradu[0][1] = (grad[lelem](1,1)*u(lelem,0)-u(lelem,1)*grad[lelem](1,0))
							/ (u(lelem,0)*u(lelem,0));
			gradu[1][0] = (grad[lelem](0,2)*u(lelem,0)-u(lelem,2)*grad[lelem](0,0))
							/ (u(lelem,0)*u(lelem,0));
			gradu[1][1] = (grad[lelem](1,2)*u(lelem,0)-u(lelem,2)*grad[lelem](1,0))
							/ (u(lelem,0)*u(lelem,0));
			
			const scalar tauw = 
				muhat*((2.0*gradu[0][0]*n[0] +(gradu[0][1]+gradu[1][0])*n[1])*n[1]
				+ ((gradu[1][0]+gradu[0][1])*n[0] + 2.0*gradu[1][1]*n[1])*(-n[0]));

			output(facecoun, NDIM+1) = 2.0*tauw;

			// add contributions to Cdp, Cdf and Cl
			
			// face normal dot free-stream direction
			const scalar ndotf = n[0]*av[0]+n[1]*av[1];
			// face normal dot "up" direction perpendicular to free stream
			const scalar ndotnf = n[0]*flownormal[0]+n[1]*flownormal[1];
			// face tangent dot free-stream direction
			const scalar tdotf = n[1]*av[0]-n[0]*av[1];

			Cdp += output(facecoun,NDIM)*ndotf*len;
			Cdf += output(facecoun,NDIM+1)*tdotf*len;
			Cl += output(facecoun,NDIM)*ndotnf*len;

			facecoun++;
		}
	}

	// Normalize drag and lift by reference area
	Cdp /= totallen; Cdf /= totallen; Cl /= totallen;

	return std::make_tuple(Cl, Cdp, Cdf);
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
FlowFV<scalar,secondOrderRequested,constVisc>::FlowFV(const UMesh2dh<scalar> *const mesh,
                                                      const FlowPhysicsConfig& pconf, 
                                                      const FlowNumericsConfig& nconf)
	: FlowFV_base<scalar>(mesh, pconf, nconf),
	jphy(pconfig.gamma, pconfig.Minf, pconfig.Tinf, pconfig.Reinf, pconfig.Pr),
	juinf(jphy.compute_freestream_state(pconfig.aoa)),
	jflux {create_const_inviscidflux<a_real>(nconfig.conv_numflux_jac, &jphy)},
		   jbcs {create_const_flowBCs<a_real>(pconf.bcconf, jphy, juinf)}
{
	if(secondOrderRequested)
		std::cout << "FlowFV: Second order solution requested.\n";
	if(constVisc)
		std::cout << " FLowFV: Using constant viscosity.\n";
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
FlowFV<scalar,secondOrderRequested,constVisc>::~FlowFV()
{
	delete jflux;
	// delete BCs
	for(auto it = jbcs.begin(); it != jbcs.end(); it++) {
		delete it->second;
	}
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
void FlowFV<scalar,secondOrderRequested,constVisc>
::compute_viscous_flux(const a_int iface,
                       const scalar *const ucell_l, const scalar *const ucell_r,
                       const amat::Array2d<scalar>& ug,
                       const GradArray<scalar,NVARS>& grads,
                       const amat::Array2d<scalar>& ul, const amat::Array2d<scalar>& ur,
                       scalar *const __restrict vflux) const
{
	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);
	const std::array<scalar,NDIM> normal = m->gnormal(iface);

	// cell-centred left and right states
	scalar uctl[NVARS], uctr[NVARS];
	// left and right gradients; zero for first order scheme
	scalar gradl[NDIM*NVARS], gradr[NDIM*NVARS];

	const scalar *in_ucr = nullptr;
	//const scalar *in_gradl = nullptr, *in_gradr = nullptr;
	
	if(iface < m->gnbface())
	{
		// boundary face
		if(secondOrderRequested)
		{
			in_ucr = &ug(iface,0);

			// Use the same gradients on both sides of a boundary face;
			// this will amount to just using the one-sided gradient for the modified average
			// gradient later.
			// NOTE: We can't take the address of a GradArray entry, so we need copies here.
			// in_gradl = &grads[lelem](0,0);
			// in_gradr = &grads[lelem](0,0);

			for(int i = 0; i < NDIM; i++)
				for(int j = 0; j < NVARS; j++) {
					gradl[i*NVARS+j] = grads[lelem](i,j);
					gradr[i*NVARS+j] = grads[lelem](i,j);
				}
		}
		else
		{
			// if second order was not requested, ghost cell values are stored in ur, not ug
			in_ucr = &ur(iface,0);
		}
	}
	else
	{
		in_ucr = ucell_r;
		if(secondOrderRequested) {
			// in_gradl = &grads[lelem](0,0);
			// in_gradr = &grads[relem](0,0);
			for(int i = 0; i < NDIM; i++)
				for(int j = 0; j < NVARS; j++) {
					gradl[i*NVARS+j] = grads[lelem](i,j);
					gradr[i*NVARS+j] = grads[relem](i,j);
				}
		}
	}

	// getPrimitive2StateAndGradients<scalar,NDIM,secondOrderRequested>(physics, ucell_l, in_ucr,
	//                                                                  in_gradl, in_gradr,
	//                                                                  uctl, uctr, gradl, gradr);

	getPrimitive2StatesAndGradients<scalar,NDIM,secondOrderRequested>(physics, ucell_l, in_ucr,
	                                                                  gradl, gradr,
	                                                                  uctl, uctr, gradl, gradr);

	if(!secondOrderRequested) {
		for(int i = 0; i < NDIM; i++)
			for(int j = 0; j < NVARS; j++) {
				gradl[i*NVARS+j] = 0;
				gradr[i*NVARS+j] = 0;
			}
	}

	scalar grad[NDIM][NVARS];
	getFaceGradient_modifiedAverage(iface, uctl, uctr, gradl, gradr, grad);

	computeViscousFlux<scalar,NDIM,NVARS,constVisc>(physics, &normal[0], grad,
	                                                &ul(iface,0), &ur(iface,0), vflux);
}

template<typename scalar, bool secondOrder, bool constVisc>
void FlowFV<scalar,secondOrder,constVisc>::computeViscousFluxJacobian(const a_int iface,
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

	jphy.getPrimitive2FromConserved(ul, upl);
	jphy.getPrimitive2FromConserved(ur, upr);

	jphy.getJacobianPrimitive2WrtConserved(ul, dupl);
	jphy.getJacobianPrimitive2WrtConserved(ur, dupr);
	
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
			jphy.getConstantViscosityCoeff() 
		:
			0.5*( jphy.getViscosityCoeffFromConserved(ul)
			+ jphy.getViscosityCoeffFromConserved(ur) );
	
	// Non-dimensional thermal conductivity
	const a_real kdiff = jphy.getThermalConductivityFromViscosity(muRe); 

	a_real dmul[NVARS], dmur[NVARS], dkdl[NVARS], dkdr[NVARS];
	for(int k = 0; k < NVARS; k++) {
		dmul[k] = 0; dmur[k] = 0; dkdl[k] = 0; dkdr[k] = 0;
	}

	if(!constVisc) {
		jphy.getJacobianSutherlandViscosityWrtConserved(ul, dmul);
		jphy.getJacobianSutherlandViscosityWrtConserved(ur, dmur);
		for(int k = 0; k < NVARS; k++) {
			dmul[k] *= 0.5;
			dmur[k] *= 0.5;
		}
		jphy.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmul, dkdl);
		jphy.getJacobianThermCondWrtConservedFromJacobianSutherViscWrtConserved(dmur, dkdr);
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
	
	jphy.getJacobianStress(muRe, dmul, grad, dgradl, stress, dstressl);
	jphy.getJacobianStress(muRe, dmur, grad, dgradr, stress, dstressr);

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

template<typename scalar, bool secondOrder, bool constVisc>
void FlowFV<scalar,secondOrder,constVisc>::computeViscousFluxApproximateJacobian(const a_int iface,
		const a_real *const ul, const a_real *const ur,
		a_real *const __restrict dvfi, a_real *const __restrict dvfj) const
{
	// compute non-dimensional viscosity and thermal conductivity
	const a_real muRe = constVisc ? 
			jphy.getConstantViscosityCoeff() 
		:
			0.5*( jphy.getViscosityCoeffFromConserved(ul)
			+ jphy.getViscosityCoeffFromConserved(ur) );
	
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

template<typename scalar, bool secondOrderRequested, bool constVisc>
StatusCode FlowFV<scalar,secondOrderRequested,constVisc>::compute_residual(const scalar *const uarr, 
		scalar *const __restrict rarr, 
		const bool gettimesteps, std::vector<a_real>& dtm) const
{
	StatusCode ierr = 0;
	amat::Array2d<scalar> integ, ug, uleft, uright;	
	integ.resize(m->gnelem(), 1);
	ug.resize(m->gnbface(),NVARS);
	uleft.resize(m->gnaface(), NVARS);
	uright.resize(m->gnaface(), NVARS);
	GradArray<scalar,NVARS> grads;

	Eigen::Map<const MVector<scalar>> u(uarr, m->gnelem(), NVARS);
	Eigen::Map<MVector<scalar>> residual(rarr, m->gnelem(), NVARS);

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
		grads.resize(m->gnelem());

		// get cell average values at ghost cells using BCs
		compute_boundary_states(uleft, ug);

		MVector<scalar> up(m->gnelem(), NVARS);

		// convert cell-centered state vectors to primitive variables
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
				physics.getConservedFromPrimitive(&ug(iface,0), &ug(iface,0));
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

	// Compute fluxes.
	/**
	 * The integral of the spectral radius of the (one-sided analytical) flux Jacobian over
	 * each face \f$ f_i \f$ is also computed and summed over for each cell \f$ K \f$:
	 * \f[
	 * \sum_{f_i \in \partial K} \int_{f_i} (|v_n| + c + \lamba_v) \mathrm{d}\gamma
	 * \f]
	 * so that time steps can be calculated for explicit time stepping and/or steady problems.
	 * Note that the reconstructed state is used to compute the spectral radius.
	 * \f$ \lambda_v \f$ is an estimate of the spectral radius of the viscous flux Jacobian, taken
	 * from \cite{blazek}.
	 */

#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int ied = 0; ied < m->gnaface(); ied++)
		{
			scalar n[NDIM];
			n[0] = m->gfacemetric(ied,0);
			n[1] = m->gfacemetric(ied,1);
			scalar len = m->gfacemetric(ied,2);
			const int lelem = m->gintfac(ied,0);
			const int relem = m->gintfac(ied,1);
			scalar fluxes[NVARS];

			inviflux->get_flux(&uleft(ied,0), &uright(ied,0), n, fluxes);

			// integrate over the face
			for(int ivar = 0; ivar < NVARS; ivar++)
					fluxes[ivar] *= len;

			if(pconfig.viscous_sim) 
			{
				// get viscous fluxes
				scalar vflux[NVARS];
				const scalar *const urt = (ied < m->gnbface()) ? nullptr : &uarr[relem*NVARS];
				compute_viscous_flux(ied, &uarr[lelem*NVARS], urt, ug, grads, uleft, uright, 
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
				const scalar ci = physics.getSoundSpeedFromConserved(&uleft(ied,0));
				const scalar cj = physics.getSoundSpeedFromConserved(&uright(ied,0));
				//calculate normal velocities
				const scalar vni = (uleft(ied,1)*n[0] +uleft(ied,2)*n[1])/uleft(ied,0);
				const scalar vnj = (uright(ied,1)*n[0] + uright(ied,2)*n[1])/uright(ied,0);

				scalar specradi = (fabs(vni)+ci)*len;
				scalar specradj = (fabs(vnj)+cj)*len;

				if(pconfig.viscous_sim) 
				{
					scalar mui, muj;
					if(constVisc) {
						mui = physics.getConstantViscosityCoeff();
						muj = physics.getConstantViscosityCoeff();
					}
					else {
						mui = physics.getViscosityCoeffFromConserved(&uleft(ied,0));
						muj = physics.getViscosityCoeffFromConserved(&uright(ied,0));
					}
					const scalar coi = std::max(4.0/(3*uleft(ied,0)), physics.g/uleft(ied,0));
					const scalar coj = std::max(4.0/(3*uright(ied,0)), physics.g/uright(ied,0));
					
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
	
	return ierr;
}

template<typename scalar, bool order2, bool constVisc>
StatusCode FlowFV<scalar,order2,constVisc>::compute_jacobian(const Vec uvec, Mat A) const
{
	StatusCode ierr = 0;

	PetscInt locnelem; const PetscScalar *uarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const std::array<a_real,NDIM> n = m->gnormal(iface);
		const a_real len = m->gfacemetric(iface,2);
		
		a_real uface[NVARS];
		Matrix<a_real,NVARS,NVARS,RowMajor> drdl;
		Matrix<a_real,NVARS,NVARS,RowMajor> left;
		Matrix<a_real,NVARS,NVARS,RowMajor> right;
		
		bcs.at(m->gintfacbtags(iface,0))->computeGhostStateAndJacobian(&uarr[lelem*NVARS], &n[0],
		                                                               uface, &drdl(0,0));
		
		jflux->get_jacobian(&uarr[lelem*NVARS], uface, &n[0], &left(0,0), &right(0,0));

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

template class FlowFV_base<a_real>;

template class FlowFV<a_real,true,true>;
template class FlowFV<a_real,false,true>;
template class FlowFV<a_real,true,false>;
template class FlowFV<a_real,false,false>;

}


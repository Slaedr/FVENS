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
#include "physics/aphysics_defs.hpp"
#include "physics/viscousphysics.hpp"
#include "abctypemap.hpp"
#include "utilities/afactory.hpp"
#include "utilities/adolcutils.hpp"
#include "utilities/mpiutils.hpp"
#include "linalg/petscutils.hpp"
#include "linalg/tracevector.hpp"
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

	gradcomp {create_const_gradientscheme<scalar,NVARS>(nconfig.gradientscheme, m, rch.getArray(),
	                                                    &rcbp(0,0))},
	lim {create_const_reconstruction<scalar,NVARS>(nconfig.reconstruction, m, rch.getArray(), &rcbp(0,0),
	                                               gr, nconfig.limiter_param)},

	bcs {create_const_flowBCs<scalar>(pconf.bcconf, physics,uinf)}

{
	std::cout << " FlowFV_base: Boundary conditions:\n";
	for(auto it = pconfig.bcconf.begin(); it != pconfig.bcconf.end(); it++) {
		std::cout << "  " << bcTypeMap.left.find(it->bc_type)->second << '\n';
	}

	// To get rid of unused function warning for ADOL-C include
	//  Should be removed once ADOL-C instantiations are compiled and working
#ifdef USE_ADOLC
	adouble x = 3.0;
	(void)getvalue<adouble>(x);
#endif
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
void FlowFV_base<scalar>::compute_boundary_states(const scalar *const ins, scalar *const gs) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		compute_boundary_state(ied,
		                       ins + (ied-m->gPhyBFaceStart())*NVARS,
		                       gs  + (ied-m->gPhyBFaceStart())*NVARS);
	}
}

template <typename scalar>
void FlowFV_base<scalar>::compute_boundary_state(const int ied,
                                                 const scalar *const ins,
                                                 scalar *const gs
                                                 ) const
{
	const std::array<scalar,NDIM> n = m->gnormal(ied);
	bcs.at(m->gbtags(ied,0))->computeGhostState(ins, &n[0], gs);
}

template <typename scalar>
void FlowFV_base<scalar>::getGradients(const MVector<scalar>& u,
                                       GradBlock_t<scalar,NDIM,NVARS> *const grads) const
{
	amat::Array2d<scalar> ug(m->gnbface(),NVARS);
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface-m->gPhyBFaceStart(),0));
	}

	const scalar *const ugp = m->gnbface() > 0 ? &ug(0,0) : nullptr;
	gradcomp->compute_gradients(u, amat::Array2dView<scalar>(ugp,m->gnbface(),NVARS), &grads[0](0,0));
}

template <typename scalar>
static inline std::array<scalar,NDIM> flowDirectionVector(const scalar aoa)
{
	std::array<scalar,NDIM> dir;
	for(int i = 0; i < NDIM; i++) dir[i] = 0;

	static_assert(NDIM == 2, "Flow direction not implemented for 3D yet");
	dir[0] = cos(aoa);
	dir[1] = sin(aoa);

	return dir;
}

template <typename scalar>
std::tuple<scalar,scalar,scalar>
FlowFV_base<scalar>::computeSurfaceData (const MVector<scalar>& u,
                                         const GradBlock_t<scalar,NDIM,NVARS> *const grad,
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
	static_assert(NDIM == 2, "surface data not implemented for 3D yet");
	scalar flownormal[NDIM]; flownormal[0] = -av[1]; flownormal[1] = av[0];

	// iterate over faces having this boundary marker
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		if(m->gbtags(iface,0) == iwbcm)
		{
			const a_int lelem = m->gintfac(iface,0);
			scalar n[NDIM];
			for(int j = 0; j < NDIM; j++)
				n[j] = m->gfacemetric(iface,j);
			const scalar len = m->gfacemetric(iface,2);

			// coords of face center
			a_int ijp[NDIM];
			for(int j = 0; j < NDIM; j++)
				ijp[j] = m->gintfac(iface,2+j);
			scalar coord[NDIM];
			for(int j = 0; j < NDIM; j++)
			{
				coord[j] = 0;
				for(int inofa = 0; inofa < m->gnnofa(iface); inofa++)
					coord[j] += m->gcoords(ijp[inofa],j);
				coord[j] /= m->gnnofa(iface);

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

			totallen += len;
			Cdp += output(facecoun,NDIM)*ndotf*len;
			Cdf += output(facecoun,NDIM+1)*tdotf*len;
			Cl += output(facecoun,NDIM)*ndotnf*len;

			facecoun++;
		}
	}

	scalar funcs[] = {Cl, Cdp, Cdf, totallen};
	mpi_all_reduce<scalar>(funcs, 4, MPI_SUM, MPI_COMM_WORLD);
	Cl = funcs[0]; Cdp = funcs[1]; Cdf = funcs[2]; totallen = funcs[3];

	// Normalize drag and lift by reference area
	Cdp /= totallen; Cdf /= totallen; Cl /= totallen;

	return std::make_tuple(Cl, Cdp, Cdf);
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
FlowFV<scalar,secondOrderRequested,constVisc>::FlowFV(const UMesh2dh<scalar> *const mesh,
                                                      const FlowPhysicsConfig& pconf,
                                                      const FlowNumericsConfig& nconf)
	: FlowFV_base<scalar>(mesh, pconf, nconf),
	jphy(pconfig.gamma, pconfig.Minf, pconfig.Tinf, pconfig.Reinf, pconfig.Pr)
{
	if(secondOrderRequested)
		std::cout << "FlowFV: Second order solution requested.\n";
	if(constVisc)
		std::cout << " FLowFV: Using constant viscosity.\n";
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
FlowFV<scalar,secondOrderRequested,constVisc>::~FlowFV()
{
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
void FlowFV<scalar,secondOrderRequested,constVisc>
::compute_viscous_flux(const scalar *const normal,
                       const scalar *const rcl, const scalar *const rcr,
                       const scalar *const ucell_l, const scalar *const ucell_r,
                       const GradBlock_t<scalar,NDIM,NVARS>& gradsl,
                       const GradBlock_t<scalar,NDIM,NVARS>& gradsr,
                       const scalar *const ul, const scalar *const ur,
                       scalar *const __restrict vflux) const
{
	// cell-centred left and right states
	scalar uctl[NVARS], uctr[NVARS];
	// left and right gradients; zero for first order scheme
	scalar gradl[NDIM*NVARS], gradr[NDIM*NVARS];

	// const scalar *in_gradl = nullptr, *in_gradr = nullptr;

	if(secondOrderRequested) {
		// in_gradl = &grads[lelem](0,0);
		// in_gradr = &grads[relem](0,0);
		for(int i = 0; i < NDIM; i++)
			for(int j = 0; j < NVARS; j++) {
				gradl[i*NVARS+j] = gradsl(i,j);
				gradr[i*NVARS+j] = gradsr(i,j);
			}
	}

	// getPrimitive2StatesAndGradients<scalar,NDIM,secondOrderRequested>(physics, ucell_l, in_ucr,
	//                                                                   in_gradl, in_gradr,
	//                                                                   uctl, uctr, gradl, gradr);

	getPrimitive2StatesAndGradients<scalar,NDIM,secondOrderRequested>(physics, ucell_l, ucell_r,
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
	getFaceGradient_modifiedAverage(rcl, rcr, uctl, uctr, gradl, gradr, grad);

	computeViscousFlux<scalar,NDIM,NVARS,constVisc>(physics, normal, grad, ul, ur, vflux);
}

template<typename scalar, bool secondOrder, bool constVisc>
void FlowFV<scalar,secondOrder,constVisc>
::compute_viscous_flux_jacobian(const a_int iface,
                                const a_real *const ul, const a_real *const ur,
                                a_real *const __restrict dvfi, a_real *const __restrict dvfj) const
{
	const amat::Array2dView<a_real> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

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

	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);

	if(iface >= m->gPhyBFaceStart() && iface < m->gPhyBFaceEnd())
		getFaceGradientAndJacobian_thinLayer(&rc(lelem,0), &rcbp(iface-m->gPhyBFaceStart()),
		                                     upl, upr, dupl, dupr, grad,
		                                     dgradl, dgradr);
	else
		getFaceGradientAndJacobian_thinLayer(&rc(lelem,0), &rc(relem,0), upl, upr, dupl, dupr, grad,
		                                     dgradl, dgradr);

	const std::array<scalar,NDIM> n = m->gnormal(iface);
	computeViscousFluxJacobian<scalar,NDIM,NVARS,constVisc>(jphy, &n[0], ul, ur, grad, dgradl, dgradr,
	                                                        dvfi, dvfj);
}

template<typename scalar, bool secondOrder, bool constVisc>
void FlowFV<scalar,secondOrder,constVisc>
::compute_viscous_flux_approximate_jacobian(const a_int iface,
                                            const a_real *const ul, const a_real *const ur,
                                            a_real *const __restrict dvfi,
                                            a_real *const __restrict dvfj) const
{
	const amat::Array2dView<a_real> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

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
void FlowFV<scalar,secondOrderRequested,constVisc>
::compute_fluxes(const scalar *const u, const scalar *const gradients,
                 const scalar *const uleft, const scalar *const uright,
                 const scalar *const ug,
                 scalar *const res) const
{
	const amat::Array2dView<a_real> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);
	const GradBlock_t<scalar,NDIM,NVARS> *const grads
		= reinterpret_cast<const GradBlock_t<scalar,NDIM,NVARS>*>(gradients);
	Eigen::Map<MVector<scalar>> residual(res, m->gnelem(), NVARS);

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

#pragma omp parallel for default(shared)
	for(a_int ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		scalar n[NDIM];
		n[0] = m->gfacemetric(ied,0);
		n[1] = m->gfacemetric(ied,1);
		const scalar len = m->gfacemetric(ied,2);
		const a_int lelem = m->gintfac(ied,0);
		const a_int relem = m->gintfac(ied,1);
		scalar fluxes[NVARS];

		inviflux->get_flux(&uleft[ied*NVARS], &uright[ied*NVARS], n, fluxes);

		// integrate over the face
		for(int ivar = 0; ivar < NVARS; ivar++)
			fluxes[ivar] *= len;

		if(pconfig.viscous_sim)
		{
			const a_int ibpface = ied - m->gPhyBFaceStart();
			const bool isPhyBoun = (ied >= m->gPhyBFaceStart() && ied < m->gPhyBFaceEnd());
			const scalar *const rcr = isPhyBoun ? &rcbp(ibpface,0) : &rc(relem,0);
			const scalar *const ucellright
				= isPhyBoun ? &ug[ibpface*NVARS] : &u[relem*NVARS];
			const GradBlock_t<scalar,NDIM,NVARS>& gradright = isPhyBoun ? grads[lelem] : grads[relem];

			scalar vflux[NVARS];
			compute_viscous_flux(n, &rc(lelem,0), rcr,
			                     &u[lelem*NVARS], ucellright, grads[lelem], gradright,
			                     &uleft[ied*NVARS], &uright[ied*NVARS], vflux);

			for(int ivar = 0; ivar < NVARS; ivar++)
				fluxes[ivar] += vflux[ivar]*len;
		}

		/// We assemble the negative of the residual ( M du/dt + r(u) = 0).
		for(int ivar = 0; ivar < NVARS; ivar++) {
#pragma omp atomic update
			residual(lelem,ivar) -= fluxes[ivar];
		}
		if(relem < m->gnelem()) {
			for(int ivar = 0; ivar < NVARS; ivar++) {
#pragma omp atomic update
				residual(relem,ivar) += fluxes[ivar];
			}
		}
	}
}

// compute max allowable time steps
template<typename scalar, bool secondOrderRequested, bool constVisc>
void FlowFV<scalar,secondOrderRequested,constVisc>
::compute_max_timestep(const amat::Array2dView<scalar> uleft,
                       const amat::Array2dView<scalar> uright,
                       a_real *const timesteps) const
{
	amat::Array2d<a_real> integ(m->gnelem(),1);
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		integ(iel) = 0.0;
	}

#pragma omp parallel for default(shared)
	for(a_int ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		scalar n[NDIM];
		n[0] = m->gfacemetric(ied,0);
		n[1] = m->gfacemetric(ied,1);
		const scalar len = m->gfacemetric(ied,2);
		const int lelem = m->gintfac(ied,0);
		const int relem = m->gintfac(ied,1);
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

#pragma omp atomic update
		integ(lelem) += getvalue<scalar>(specradi);

		if(relem < m->gnelem()) {
#pragma omp atomic update
			integ(relem) += getvalue<scalar>(specradj);
		}
	}

#ifdef USE_ADOLC
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd default(shared)
#endif
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		timesteps[iel] = getvalue<scalar>(m->garea(iel))/integ(iel);
	}
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
StatusCode
FlowFV<scalar,secondOrderRequested,constVisc>::compute_residual(const Vec uvec,
                                                                Vec rvec,
                                                                const bool gettimesteps,
                                                                Vec timesteps) const
{
	StatusCode ierr = 0;
	GradBlock_t<scalar,NDIM,NVARS>* grads = nullptr;
	L2TraceVector<scalar,NVARS> uface(*m);

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());

	const ConstVecHandler<scalar> uvh(uvec);
	const scalar *const uarr = uvh.getArray();
	Eigen::Map<const MVector<scalar>> u(uarr, m->gnelem(), NVARS);

	{
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),NVARS);
		// first, set cell-centered values of boundary cells as left-side values of boundary faces
#pragma omp parallel for default(shared)
		for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		{
			const a_int ielem = m->gintfac(ied,0);
			for(int ivar = 0; ivar < NVARS; ivar++)
				uleft(ied,ivar) = u(ielem,ivar);
		}
	}

	// cell-centred ghost cell values corresponding to physical boundaries
	scalar *ubcell = nullptr;
	if(m->gnbface() > 0)
		ubcell = new scalar[m->gnbface()*NVARS];

	if(secondOrderRequested)
	{
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),NVARS);
		amat::Array2dMutableView<scalar> uright(uface.getLocalArrayRight(), m->gnaface(),NVARS);

		// for storing cell-centred gradients at interior cells and ghost cells
		grads = new GradBlock_t<scalar,NDIM,NVARS>[m->gnelem()];

		// get cell average values at ghost cells using BCs for reconstruction
		compute_boundary_states(&uleft(m->gPhyBFaceStart(),0), &uright(m->gPhyBFaceStart(),0));

		MVector<scalar> up(m->gnelem(), NVARS);

		// convert cell-centered state vectors to primitive variables
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
			{
				// Save ghost cell-centred physical boundary (conserved) values for later
				for(int j = 0; j < NVARS; j++)
					ubcell[(iface-m->gPhyBFaceStart())*NVARS+j] = uright(iface,j);

				// convert boundary values to primitive
				physics.getPrimitiveFromConserved(&uright(iface,0), &uright(iface,0));
			}

#pragma omp for
			for(a_int iel = 0; iel < m->gnelem(); iel++)
				physics.getPrimitiveFromConserved(&uarr[iel*NVARS], &up(iel,0));
		}

		const amat::Array2dView<scalar> ug(&uright(m->gPhyBFaceStart(),0),m->gnbface(),NVARS);

		// reconstruct
		gradcomp->compute_gradients(up, ug, &grads[0](0,0));
		lim->compute_face_values(up, ug, &grads[0](0,0), uleft, uright);

		// Convert face values back to conserved variables - gradients stay primitive.
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(a_int iface = m->gDomFaceStart(); iface < m->gDomFaceEnd(); iface++)
			{
				physics.getConservedFromPrimitive(&uleft(iface,0), &uleft(iface,0));
				physics.getConservedFromPrimitive(&uright(iface,0), &uright(iface,0));
			}
#pragma omp for
			for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
			{
				physics.getConservedFromPrimitive(uface.getLocalArrayLeft()+iface*NVARS,
				                                  uface.getLocalArrayLeft()+iface*NVARS);
			}
		}
	}
	else
	{
		// if order is 1, set the face data same as cell-centred data for all faces

		// set both left and right states for all interior and connectivity faces
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),NVARS);
		amat::Array2dMutableView<scalar> uright(uface.getLocalArrayRight(), m->gnaface(),NVARS);
#pragma omp parallel for default(shared)
		for(a_int ied = m->gDomFaceStart(); ied < m->gDomFaceEnd(); ied++)
		{
			const a_int ielem = m->gintfac(ied,0);
			const a_int jelem = m->gintfac(ied,1);
			for(int ivar = 0; ivar < NVARS; ivar++)
			{
				uleft(ied,ivar) = u(ielem,ivar);
				uright(ied,ivar) = u(jelem,ivar);
			}
		}
	}

	// get right (ghost) state at boundary faces for computing fluxes
	compute_boundary_states(uface.getLocalArrayLeft()+m->gPhyBFaceStart()*NVARS,
	                        uface.getLocalArrayRight()+m->gPhyBFaceStart()*NVARS);

	MutableVecHandler<scalar> rvh(rvec);
	scalar *const rarr = rvh.getArray();
	// Depending on whether we want a 2nd order solution, we use the correct array for phy. boun.
	//  ghost cells
	const scalar *const ug_pb = secondOrderRequested ?
		ubcell : uface.getLocalArrayRight()+m->gPhyBFaceStart()*NVARS;

	compute_fluxes(uarr, &grads[0](0,0), uface.getLocalArrayLeft(), uface.getLocalArrayRight(),
	               ug_pb, rarr);

	if(gettimesteps)
	{
		MutableVecHandler<a_real> dtvh(timesteps);
		a_real *const dtm = dtvh.getArray();
		compute_max_timestep(amat::Array2dView<scalar>(uface.getLocalArrayLeft(),m->gnaface(),NVARS),
		                     amat::Array2dView<scalar>(uface.getLocalArrayRight(),m->gnaface(),NVARS),
		                     dtm);
	}

	delete [] grads;
	delete [] ubcell;
	return ierr;
}

template<typename scalar, bool order2, bool constVisc>
void FlowFV<scalar,order2,constVisc>
::compute_local_jacobian_interior(const a_int iface,
                                  const a_real *const ul, const a_real *const ur,
                                  Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& L,
                                  Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& U) const
{
	assert(iface >= m->gDomFaceStart());
	assert(iface < m->gDomFaceEnd());

	const std::array<a_real,NDIM> n = m->gnormal(iface);
	const a_real len = m->gfacemetric(iface,2);

	// NOTE: the values of L and U get REPLACED here, not added to
	inviflux->get_jacobian(ul, ur, &n[0], &L(0,0), &U(0,0));

	if(pconfig.viscous_sim) {
		// Vec rclocal;
		// int ierr = VecGhostGetLocalForm(rcvec, &rclocal); petsc_throw(ierr, "Could not get rc local");
		// const a_real *const rcloc = getVecAsReadOnlyArray(rclocal);
		//compute_viscous_flux_approximate_jacobian(iface, &uarr[lelem*NVARS], &uarr[relem*NVARS],
		//		&L(0,0), &U(0,0));
		compute_viscous_flux_jacobian(iface, ul, ur, &L(0,0), &U(0,0));

		// ierr = VecGhostRestoreLocalForm(rcvec, &rclocal); petsc_throw(ierr, "Could not restore rc local");
	}

	L *= len; U *= len;
}

template<typename scalar, bool order2, bool constVisc>
void FlowFV<scalar,order2,constVisc>
::compute_local_jacobian_boundary(const a_int iface,
                                  const a_real *const ul,
                                  Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor>& left) const
{
	assert(iface >= m->gPhyBFaceStart());
	assert(iface < m->gPhyBFaceEnd());

	const std::array<a_real,NDIM> n = m->gnormal(iface);
	const a_real len = m->gfacemetric(iface,2);

	a_real uface[NVARS];
	Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor> drdl;
	Eigen::Matrix<a_real,NVARS,NVARS,Eigen::RowMajor> right;

	bcs.at(m->gbtags(iface,0))->computeGhostStateAndJacobian(ul, &n[0], uface, &drdl(0,0));

	inviflux->get_jacobian(ul, uface, &n[0], &left(0,0), &right(0,0));

	if(pconfig.viscous_sim) {
		//compute_viscous_flux_approximate_jacobian(iface, &uarr[lelem*NVARS], uface,
		//		&left(0,0), &right(0,0));
		compute_viscous_flux_jacobian(iface, ul, uface, &left(0,0), &right(0,0));
	}

	/* The actual derivative is  dF/dl  +  dF/dr * dr/dl.
	 * We actually need to subtract dF/dr from dF/dl because the inviscid numerical flux
	 * computation returns the negative of dF/dl but positive dF/dr. The latter was done to
	 * get correct signs for lower and upper off-diagonal blocks.
	 *
	 * Integrate the results over the face --- NO -> and negate, as -ve of L is added to D
	 */
	left = len*(left - right*drdl);
}

template class FlowFV_base<a_real>;

template class FlowFV<a_real,true,true>;
template class FlowFV<a_real,false,true>;
template class FlowFV<a_real,true,false>;
template class FlowFV<a_real,false,false>;

//#ifdef USE_ADOLC
//template class FlowFV_base<adouble>;

//template class FlowFV<adouble,true,true>;
//template class FlowFV<adouble,false,true>;
//template class FlowFV<adouble,true,false>;
//template class FlowFV<adouble,false,false>;
//#endif

}


#include <iostream>
#include "convdiff.hpp"
#include "utilities/afactory.hpp"
#include "utilities/aerrorhandling.hpp"
#include "linalg/petscutils.hpp"
#include "linalg/alinalg.hpp"

namespace fvens {
namespace convdiff {

template<int nvars>
ConvectionDiffusion<nvars>::ConvectionDiffusion(const UMesh<freal,NDIM> *const mesh,
	    const PhysicsConfig& pc, const NumericsConfig& nc)
	: Spatial<freal,nvars>(mesh), pconfig{pc}, nconfig{nc}, h(mesh->getCellSizes()),
	gradcomp (create_const_gradientscheme<freal,nvars>(nc.gradientscheme, m, rch.getArray(), rcbptr)),
	lim (create_const_reconstruction<freal,nvars>(
			 nc.reconstruction, m, rch.getArray(), rcbptr, gr, nc.limiter_param)),
    bcs {create_const_BCs<nvars>(pconfig.bcconf)}, uface(*mesh)
{
	if(nconfig.order2) {
		std::cout << " ConvectionDiffusion: Second order solution requested.\n";
		// for storing cell-centred gradients at interior cells and ghost cells
		int ierr = createGhostedSystemVector(m, nvars*NDIM, &gradvec);
		fvens_throw(ierr, "Could not create storage for gradients!");
	}
}

template<int nvars>
ConvectionDiffusion<nvars>::~ConvectionDiffusion()
{
	int ierr = VecDestroy(&gradvec);
	if(ierr) {
		std::cout << "Gradient vector could not be destroyed!" << std::endl;
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>::compute_advective_flux(const freal *const uleft,
														const freal *const urght,
														const freal *const normal,
													   freal *const __restrict__ fluxes) const
{
	for(int ivar = 0; ivar < nvars; ivar++)
	{
		freal normvel = 0;
		for(int idim = 0; idim < NDIM; idim++) {
			normvel += pconfig.conv_velocity[idim] * normal[idim];
		}
		// upwind convective flux
		if(normvel <= 0) {
			fluxes[ivar] = normvel * urght[ivar];
		} else {
			fluxes[ivar] = normvel * uleft[ivar];
		}
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>
::compute_viscous_flux(const freal *const normal,
                       const freal *const rcl, const freal *const rcr,
                       const freal *const ucell_l, const freal *const ucell_r,
                       const GradBlock_t<freal,NDIM,nvars>& gradsl,
                       const GradBlock_t<freal,NDIM,nvars>& gradsr,
                       freal *const __restrict__ vflux) const
{
	freal gradl[NDIM*nvars], gradr[NDIM*nvars];
	if(nconfig.order2) {
		for(int idim = 0; idim < NDIM; idim++) {
			for(int j = 0; j < nvars; j++) {
				gradl[idim*nvars+j] = gradsl(idim,j);
				gradr[idim*nvars+j] = gradsr(idim,j);
			}
		}
	} else {
		for(int idim = 0; idim < NDIM; idim++) {
			for(int j = 0; j < nvars; j++) {
				gradl[idim*nvars+j] = 0;
				gradr[idim*nvars+j] = 0;
			}
		}
	}

	freal gradf[NDIM][nvars];
	getFaceGradient_modifiedAverage(rcl, rcr, ucell_l, ucell_r, gradl, gradr, gradf);

	for(int ivar = 0; ivar < nvars; ivar++)
	{
		// compute nu*(-grad u . n) * l
		freal vfluxi = 0;
		for(int idim = 0; idim < NDIM; idim++) {
			vfluxi += gradf[idim][ivar]*normal[idim];
		}
		vflux[ivar] = vfluxi * (-1.0/pconfig.peclet);
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>
::compute_fluxes(const freal *const u, const freal *const gradients,
                 const freal *const uleft, const freal *const uright,
                 const freal *const ug,
                 freal *const res) const
{
	const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);
	const GradBlock_t<freal,NDIM,nvars> *const grads
		= reinterpret_cast<const GradBlock_t<freal,NDIM,nvars>*>(gradients);

	/* Note that we don't need access to residuals of connectivity ghost cells. Each subdomain is
	 * responsible only for residuals in its own cells while fluxes across connectivity faces are
	 * computed twice - once by each subdomain.
	 */
	Eigen::Map<MVector<freal>> residual(res, m->gnelem(), nvars);

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
	for(fint ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		const std::array<freal,NDIM> n = m->gnormal(ied);
		const freal len = m->gfacemetric(ied,NDIM);
		const fint lelem = m->gintfac(ied,0);
		const fint relem = m->gintfac(ied,1);
		freal fluxes[nvars];

		//inviflux->get_flux(&uleft[ied*nvars], &uright[ied*nvars], &n[0], fluxes);
		compute_advective_flux(&uleft[ied*nvars], &uright[ied*nvars], &n[0], fluxes);

		if(pconfig.viscous_sim)
		{
			const fint ibpface = ied - m->gPhyBFaceStart();
			const bool isPhyBoun = (ied >= m->gPhyBFaceStart() && ied < m->gPhyBFaceEnd());
			const freal *const rcr = isPhyBoun ? &rcbp(ibpface,0) : &rc(relem,0);
			const freal *const ucellright
				= isPhyBoun ? &ug[ibpface*nvars] : &u[relem*nvars];
			const GradBlock_t<freal,NDIM,nvars>& gradright = isPhyBoun ? grads[lelem] : grads[relem];

			freal vflux[nvars];
			compute_viscous_flux(&n[0], &rc(lelem,0), rcr,
			                     &u[lelem*nvars], ucellright, grads[lelem], gradright, vflux);

			for(int ivar = 0; ivar < nvars; ivar++)
				fluxes[ivar] += vflux[ivar];
		}

		// integrate over the face
		for(int ivar = 0; ivar < nvars; ivar++)
			fluxes[ivar] *= len;

		/// We assemble the negative of the residual ( M du/dt + r(u) = 0).
		for(int ivar = 0; ivar < nvars; ivar++) {
#pragma omp atomic update
			residual(lelem,ivar) -= fluxes[ivar];
		}
		if(relem < m->gnelem()) {
			for(int ivar = 0; ivar < nvars; ivar++) {
#pragma omp atomic update
				residual(relem,ivar) += fluxes[ivar];
			}
		}
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>
::compute_max_timestep(const amat::Array2dView<freal> uleft,
                       const amat::Array2dView<freal> uright,
                       freal *const timesteps) const
{
	amat::Array2d<freal> integ(m->gnelem(),1);
#pragma omp parallel for simd default(shared)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		integ(iel) = 0.0;
	}

#pragma omp parallel for default(shared)
	for(fint ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		const std::array<freal,NDIM> n = m->gnormal(ied);
		const freal len = m->gfacemetric(ied,NDIM);
		const int lelem = m->gintfac(ied,0);
		const int relem = m->gintfac(ied,1);
		//calculate normal velocity
		const freal vn = dimDotProduct(&pconfig.conv_velocity[0],&n[0]);

		freal specradi = fabs(vn)*len;
		freal specradj = fabs(vn)*len;

		if(pconfig.viscous_sim)
		{
			specradi += 1.0/pconfig.peclet * len * len / m->garea(lelem);
			if(relem < m->gnelem()) {
				specradj += 1.0/pconfig.peclet * len * len / m->garea(relem);
			}
		}

#pragma omp atomic update
		integ(lelem) += specradi;

		if(relem < m->gnelem()) {
#pragma omp atomic update
			integ(relem) += specradj;
		}
	}

#pragma omp parallel for simd default(shared)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		timesteps[iel] = m->garea(iel)/integ(iel);
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>::compute_boundary_states(const freal *const ins, freal *const gs) const
{
#pragma omp parallel for default(shared)
	for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const std::array<freal,NDIM> n = m->gnormal(ied);
		bcs.at(m->gbtags(ied,0))->computeGhostState(ins, &n[0], gs);
	}
}

template<int nvars>
StatusCode ConvectionDiffusion<nvars>::compute_residual(const Vec uvec, Vec rvec,
                                                const bool gettimesteps, Vec timesteps) const
{
	using scalar = freal;
	StatusCode ierr = 0;
	//const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	const ConstGhostedVecHandler<scalar> uvh(uvec);
	const scalar *const uarr = uvh.getArray();
	Eigen::Map<const MVector<scalar>> u(uarr, m->gnelem()+m->gnConnFace(), nvars);

	{
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),nvars);
		// first, set cell-centered values of boundary cells as left-side values of boundary faces
#pragma omp parallel for default(shared)
		for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);
			for(int ivar = 0; ivar < nvars; ivar++)
				uleft(ied,ivar) = u(ielem,ivar);
		}
	}

	// cell-centred ghost cell values corresponding to physical boundaries
	scalar *ubcell = nullptr;
	if(m->gnbface() > 0)
		ubcell = new scalar[m->gnbface()*nvars];

	if(nconfig.order2)
	{
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),nvars);
		amat::Array2dMutableView<scalar> uright(uface.getLocalArrayRight(), m->gnaface(),nvars);

		// get cell average values at ghost cells using BCs for reconstruction
		compute_boundary_states(&uleft(m->gPhyBFaceStart(),0), &uright(m->gPhyBFaceStart(),0));

		// convert cell-centered state vectors to primitive variables
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
			{
				// Save ghost cell-centred physical boundary  values for later
				for(int j = 0; j < nvars; j++) {
					ubcell[(iface-m->gPhyBFaceStart())*nvars+j] = uright(iface,j);
				}
			}
		}

		const amat::Array2dView<scalar> ug(&uright(m->gPhyBFaceStart(),0),m->gnbface(),nvars);

		{
			amat::Array2dView<scalar> ura(uarr, m->gnelem()+m->gnConnFace(),nvars);
			MutableGhostedVecHandler<scalar> gradh(gradvec);
			gradcomp->compute_gradients(ura, ug, gradh.getArray());
		}

		// In case of WENO reconstruction, we need gradients at conn ghost cells immediately
		if(nconfig.reconstruction == "WENO")
		{
			ierr = VecGhostUpdateBegin(gradvec, INSERT_VALUES, SCATTER_FORWARD);
			CHKERRQ(ierr);
			ierr = VecGhostUpdateEnd(gradvec, INSERT_VALUES, SCATTER_FORWARD);
			CHKERRQ(ierr);
		}

		// reconstruct
		{
			const ConstGhostedVecHandler<scalar> gradh(gradvec);
			lim->compute_face_values(u, ug, gradh.getArray(), uleft, uright);
		}

		if(nconfig.reconstruction != "WENO")
		{
			ierr = VecGhostUpdateBegin(gradvec, INSERT_VALUES, SCATTER_FORWARD);
			CHKERRQ(ierr);
		}

		uface.updateSharedFacesBegin();
	}
	else
	{
		// if order is 1, set the face data same as cell-centred data for all faces

		// set both left and right states for all interior and connectivity faces
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),nvars);
		amat::Array2dMutableView<scalar> uright(uface.getLocalArrayRight(), m->gnaface(),nvars);
#pragma omp parallel for default(shared)
		for(fint ied = m->gDomFaceStart(); ied < m->gDomFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);
			const fint jelem = m->gintfac(ied,1);
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				uleft(ied,ivar) = u(ielem,ivar);
				uright(ied,ivar) = u(jelem,ivar);
			}
		}
	}

	// get right (ghost) state at boundary faces for computing fluxes
	compute_boundary_states(uface.getLocalArrayLeft()+m->gPhyBFaceStart()*nvars,
	                        uface.getLocalArrayRight()+m->gPhyBFaceStart()*nvars);

	if(nconfig.order2)
	{
		uface.updateSharedFacesEnd();

		if(nconfig.reconstruction != "WENO")
		{
			ierr = VecGhostUpdateEnd(gradvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
		}
	}

	ConstGhostedVecHandler<scalar> gradh;
	if(nconfig.order2)
		gradh.setVec(gradvec);
	const scalar *const gradarray = nconfig.order2 ? gradh.getArray() : nullptr;

	MutableVecHandler<scalar> rvh(rvec);
	scalar *const rarr = rvh.getArray();
	// Depending on whether we want a 2nd order solution, we use the correct array for phy. boun.
	//  ghost cells
	const scalar *const ug_pb = nconfig.order2 ?
		ubcell : uface.getLocalArrayRight()+m->gPhyBFaceStart()*nvars;

	compute_fluxes(uarr, gradarray, uface.getLocalArrayLeft(), uface.getLocalArrayRight(),
	               ug_pb, rarr);

	{
		MutableVecHandler<freal> rvh(rvec);
		freal *const rarr = rvh.getArray();
		Eigen::Map<MVector<freal>> residual(rarr, m->gnelem(), nvars);
		const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

#pragma omp parallel for default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			// add source term
			freal sourceterm[nvars];
			pconfig.source(&rc(iel,0), 0, &uarr[iel*nvars], sourceterm);
			for(int ivar = 0; ivar < nvars; ivar++)
				residual(iel,ivar) += sourceterm[ivar]*m->garea(iel);
		}
	}

	if(gettimesteps)
	{
		MutableVecHandler<freal> dtvh(timesteps);
		freal *const dtm = dtvh.getArray();
		compute_max_timestep(amat::Array2dView<scalar>(uface.getLocalArrayLeft(),m->gnaface(),nvars),
		                     amat::Array2dView<scalar>(uface.getLocalArrayRight(),m->gnaface(),nvars),
		                     dtm);
	}

	delete [] ubcell;
	return ierr;
}

template<int nvars>
void ConvectionDiffusion<nvars>
::compute_local_jacobian_interior(const fint iface,
                                  const freal *const ul, const freal *const ur,
                                  Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L,
                                  Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& U) const
{
	const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);
	const fint lelem = m->gintfac(iface,0);
	const fint relem = m->gintfac(iface,1);
	const freal len = m->gfacemetric(iface,2);

	L = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();
	U = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();

	for(int ivar = 0; ivar < nvars; ivar++)
	{
		freal normvel = 0;
		for(int idim = 0; idim < NDIM; idim++) {
			normvel += pconfig.conv_velocity[idim] * m->gfacemetric(iface, idim);
		}
		// upwind convective flux
		if(normvel <= 0) {
			//fluxes[ivar] = normvel * uarr[relem*nvars + ivar];
			U(ivar, ivar) = -normvel;
		} else {
			//fluxes[ivar] = normvel * uarr[lelem*nvars + ivar];
			L(ivar, ivar) = normvel;
		}
	}


	if(pconfig.viscous_sim) {
		freal du[nvars*nvars];
		for(int i = 0; i < nvars; i++) {
			for(int j = 0; j < nvars; j++)
				du[i*nvars+j] = 0;
			du[i*nvars+i] = 1.0;
		}

		freal grad[NDIM][nvars], dgradl[NDIM][nvars][nvars], dgradr[NDIM][nvars][nvars];

		// Compute the face gradient Jacobian; we don't actually need the gradient, however..
		getFaceGradientAndJacobian_thinLayer(&rc(lelem), &rc(relem), ul, ur, du, du, grad, dgradl, dgradr);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(d(-grad u)/du_l . n) * l
			// The viscous Jacobian is symmetric
			freal x = 0;
			for(int idim = 0; idim < NDIM; idim++) {
				x += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
			}
			L[ivar*nvars+ivar] += x * 1.0/pconfig.peclet * len;
			U[ivar*nvars+ivar] += x * 1.0/pconfig.peclet * len;
		}
	}
}

template<int nvars>
void ConvectionDiffusion<nvars>
::compute_local_jacobian_boundary(const fint iface,
                                  const freal *const ul,
                                  Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>& L) const
{
	const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);
	const fint lelem = m->gintfac(iface,0);
	const fint ibpface = iface - m->gPhyBFaceStart();
	const std::array<freal,NDIM> n = m->gnormal(iface);
	const freal len = m->gfacemetric(iface,2);

	freal uface[nvars];
	Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor> drdl;
	Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor> right;

	// Compute ghost state Jacobian
	bcs.at(m->gbtags(iface,0))->computeGhostStateAndJacobian(ul, &n[0], uface, &drdl(0,0));

	L = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();

	for(int ivar = 0; ivar < nvars; ivar++)
	{
		freal normvel = 0;
		for(int idim = 0; idim < NDIM; idim++) {
			normvel += pconfig.conv_velocity[idim] * m->gfacemetric(iface, idim);
		}
		// upwind convective flux
		if(normvel > 0) {
			//fluxes[ivar] = normvel * uarr[lelem*nvars + ivar];
			L(ivar, ivar) = normvel;
		}
	}

	if(pconfig.viscous_sim) {
		freal du[nvars*nvars];
		for(int i = 0; i < nvars; i++) {
			for(int j = 0; j < nvars; j++)
				du[i*nvars+j] = 0;
			du[i*nvars+i] = 1.0;
		}

		freal grad[NDIM][nvars], dgradl[NDIM][nvars][nvars], dgradr[NDIM][nvars][nvars];

		// Compute the face gradient and its Jacobian; we don't actually need the gradient, however
		getFaceGradientAndJacobian_thinLayer(&rc(lelem), &rcbp(ibpface), ul, ul, du, du,
											grad, dgradl, dgradr);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(d(-grad u)/du_l . n) * l
			for(int idim = 0; idim < NDIM; idim++)
				L(ivar,ivar) += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim) * 1.0/pconfig.peclet * len;
		}
	}

	/* The actual derivative is  dF/dl  +  dF/dr * dr/dl.
	 * We actually need to subtract dF/dr from dF/dl because the inviscid numerical flux
	 * computation returns the negative of dF/dl but positive dF/dr. The latter was done to
	 * get correct signs for lower and upper off-diagonal blocks.
	 *
	 * Integrate the results over the face --- NO -> and negate, as -ve of L is added to D
	 */
	L = len*(L - right*drdl);
}

// template instantiations

//CHANGE HERE
template class ConvectionDiffusion<1>;

}
}

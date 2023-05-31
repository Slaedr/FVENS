#include <iostream>
#include "convdiff.hpp"
#include "utilities/afactory.hpp"
#include "linalg/petscutils.hpp"
#include "linalg/alinalg.hpp"

namespace fvens {
namespace convdiff {

template<int nvars>
ConvectionDiffusion<nvars>::ConvectionDiffusion(const UMesh<freal,NDIM> *const mesh,
	    const PhysicsConfig& pc, const NumericsConfig& nc)
	: Spatial<freal,nvars>(mesh), pconfig{pc}, nconfig{nc}, h(mesh->getCellSizes()),
	gradcomp {create_const_gradientscheme<freal,nvars>(nc.gradientscheme, m, rch.getArray(), rcbptr)},
	uface(*mesh)
{ }

template<int nvars>
ConvectionDiffusion<nvars>::~ConvectionDiffusion()
{
	delete gradcomp;
}

template<int nvars>
void ConvectionDiffusion<nvars>::compute_flux_interior(const fint iface,
													   const amat::Array2dView<freal>& rc,
													   const freal *const uarr,
													   const GradBlock_t<freal,NDIM,nvars> *const grads,
													   amat::Array2dMutableView<freal>& residual) const
{
	const fint lelem = m->gintfac(iface,0);
	const fint relem = m->gintfac(iface,1);
	const freal len = m->gfacemetric(iface,2);

	freal gradl[NDIM*nvars], gradr[NDIM*nvars];
	for(int ivar = 0; ivar < nvars; ivar++) {
		for(int idim = 0; idim < NDIM; idim++) {
			gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
			gradr[idim*nvars+ivar] = grads[relem](idim,ivar);
		}
	}

	freal gradf[NDIM][nvars];
	getFaceGradient_modifiedAverage
		(&rc(lelem,0), &rc(relem,0), &uarr[lelem*nvars], &uarr[relem*nvars],
		 gradl, gradr, gradf);

	for(int ivar = 0; ivar < nvars; ivar++)
	{
		// compute nu*(-grad u . n) * l
		freal flux = 0;
		for(int idim = 0; idim < NDIM; idim++)
			flux += gradf[idim][ivar]*m->gfacemetric(iface,idim);
		flux *= (-1.0/pconfig.peclet*len);

		/// We assemble the negative of the residual r in 'M du/dt + r(u) = 0'
#pragma omp atomic
		residual(lelem,ivar) -= flux;

		if(relem < m->gnelem()) {
#pragma omp atomic
			residual(relem,ivar) += flux;
		}
	}
}

template<typename scalar, bool secondOrderRequested, bool constVisc>
void FlowFV<scalar,secondOrderRequested,constVisc>
::compute_max_timestep(const amat::Array2dView<scalar> uleft,
                       const amat::Array2dView<scalar> uright,
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
		const std::array<scalar,NDIM> n = m->gnormal(ied);
		const scalar len = m->gfacemetric(ied,NDIM);
		const int lelem = m->gintfac(ied,0);
		const int relem = m->gintfac(ied,1);
		//calculate normal velocity
		const scalar vn = dimDotProduct(&pconfig.conv_velocity[0],&n[0]);

		scalar specradi = fabs(vn)*len;
		scalar specradj = fabs(vn)*len;

		if(pconfig.viscous_sim)
		{
			const scalar len_scale_left = m->garea(lelem) / len;
			specradi += 1.0/pconfig.peclet / len_scale_left;
			if(relem < m->gnelem()) {
				const scalar len_scale_rght = m->garea(relem) / len;
				specradj += 1.0/pconfig.peclet / len_scale_rght;
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
StatusCode ConvectionDiffusion<nvars>::compute_residual(const Vec uvec, Vec rvec,
                                                const bool gettimesteps, Vec timesteps) const
{
	StatusCode ierr = 0;
	//const int mpirank = get_mpi_rank(PETSC_COMM_WORLD);

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % NVARS == 0);
	locnelem /= NVARS;
	assert(locnelem == m->gnelem());

	const ConstGhostedVecHandler<scalar> uvh(uvec);
	const scalar *const uarr = uvh.getArray();
	Eigen::Map<const MVector<scalar>> u(uarr, m->gnelem()+m->gnConnFace(), NVARS);

	{
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),NVARS);
		// first, set cell-centered values of boundary cells as left-side values of boundary faces
#pragma omp parallel for default(shared)
		for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);
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

		// get cell average values at ghost cells using BCs for reconstruction
		compute_boundary_states(&uleft(m->gPhyBFaceStart(),0), &uright(m->gPhyBFaceStart(),0));

		// convert cell-centered state vectors to primitive variables
#pragma omp parallel default(shared)
		{
#pragma omp for
			for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
			{
				// Save ghost cell-centred physical boundary  values for later
				for(int j = 0; j < NVARS; j++) {
					ubcell[(iface-m->gPhyBFaceStart())*NVARS+j] = uright(iface,j);
				}
			}
		}

		const amat::Array2dView<scalar> ug(&uright(m->gPhyBFaceStart(),0),m->gnbface(),NVARS);

		{
			amat::Array2dView<scalar> ura(uarr, m->gnelem()+m->gnConnFace(),NVARS);
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
		amat::Array2dMutableView<scalar> uleft(uface.getLocalArrayLeft(), m->gnaface(),NVARS);
		amat::Array2dMutableView<scalar> uright(uface.getLocalArrayRight(), m->gnaface(),NVARS);
#pragma omp parallel for default(shared)
		for(fint ied = m->gDomFaceStart(); ied < m->gDomFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);
			const fint jelem = m->gintfac(ied,1);
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

	if(secondOrderRequested)
	{
		uface.updateSharedFacesEnd();

		if(nconfig.reconstruction != "WENO")
		{
			ierr = VecGhostUpdateEnd(gradvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
		}
	}

	ConstGhostedVecHandler<scalar> gradh;
	if(secondOrderRequested)
		gradh.setVec(gradvec);
	const scalar *const gradarray = secondOrderRequested ? gradh.getArray() : nullptr;

	MutableVecHandler<scalar> rvh(rvec);
	scalar *const rarr = rvh.getArray();
	// Depending on whether we want a 2nd order solution, we use the correct array for phy. boun.
	//  ghost cells
	const scalar *const ug_pb = secondOrderRequested ?
		ubcell : uface.getLocalArrayRight()+m->gPhyBFaceStart()*NVARS;

	compute_fluxes(uarr, gradarray, uface.getLocalArrayLeft(), uface.getLocalArrayRight(),
	               ug_pb, rarr);

	if(gettimesteps)
	{
		MutableVecHandler<freal> dtvh(timesteps);
		freal *const dtm = dtvh.getArray();
		compute_max_timestep(amat::Array2dView<scalar>(uface.getLocalArrayLeft(),m->gnaface(),NVARS),
		                     amat::Array2dView<scalar>(uface.getLocalArrayRight(),m->gnaface(),NVARS),
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

	freal du[nvars*nvars];
	for(int i = 0; i < nvars; i++) {
		for(int j = 0; j < nvars; j++)
			du[i*nvars+j] = 0;
		du[i*nvars+i] = 1.0;
	}

	freal grad[NDIM][nvars], dgradl[NDIM][nvars][nvars], dgradr[NDIM][nvars][nvars];

	// Compute the face gradient Jacobian; we don't actually need the gradient, however..
	getFaceGradientAndJacobian_thinLayer(&rc(lelem), &rc(relem), ul, ur, du, du, grad, dgradl, dgradr);

	L = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();
	U = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();
	for(int ivar = 0; ivar < nvars; ivar++)
	{
		// compute nu*(d(-grad u)/du_l . n) * l
		for(int idim = 0; idim < NDIM; idim++)
			L[ivar*nvars+ivar] += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
		L[ivar*nvars+ivar] *= (diffusivity*len);
	}

	// The Jacobian is symmetric
	U = L;
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
	const freal len = m->gfacemetric(iface,2);

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

	L = Eigen::Matrix<freal,nvars,nvars,Eigen::RowMajor>::Zero();
	for(int ivar = 0; ivar < nvars; ivar++)
	{
		// compute nu*(d(-grad u)/du_l . n) * l
		for(int idim = 0; idim < NDIM; idim++)
			L(ivar,ivar) += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
		L(ivar,ivar) *= (diffusivity*len);
	}
}

// template instantiations

//CHANGE HERE
template class ConvectionDiffusion<1>;

}
}

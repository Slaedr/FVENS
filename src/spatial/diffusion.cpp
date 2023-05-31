#include <iostream>
#include "diffusion.hpp"
#include "utilities/afactory.hpp"
#include "linalg/petscutils.hpp"
#include "linalg/alinalg.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template<int nvars>
Diffusion<nvars>::Diffusion(const UMesh<freal,NDIM> *const mesh,
                            const freal diffcoeff, const freal bvalue,
                            std::function <
								void(const freal *const, const freal, const freal *const,
								     freal *const)
                            > sourcefunc)
	: Spatial<freal,nvars>(mesh), diffusivity{diffcoeff}, bval{bvalue}, source(sourcefunc), h(mesh->getCellSizes())
{ }

template<int nvars>
Diffusion<nvars>::~Diffusion()
{ }

template<int nvars>
void Diffusion<nvars>::compute_boundary_states(const freal *const instates,
                                               freal *const bounstates) const
{
	for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		compute_boundary_state(ied,
		                       &instates [(ied-m->gPhyBFaceStart())*nvars],
		                       &bounstates[(ied-m->gPhyBFaceStart())*nvars]);
}

template<int nvars>
DiffusionMA<nvars>
::DiffusionMA(const UMesh<freal,NDIM> *const mesh,
              const freal diffcoeff, const freal bvalue,
              std::function<void(const freal *const,const freal,const freal *const,freal *const)> sf,
              const std::string grad_scheme)
	: Diffusion<nvars>(mesh, diffcoeff, bvalue, sf),
	gradcomp {create_const_gradientscheme<freal,nvars>(grad_scheme, m, rch.getArray(), rcbptr)}
{ }

template<int nvars>
DiffusionMA<nvars>::~DiffusionMA()
{
	delete gradcomp;
}

template<int nvars>
inline void DiffusionMA<nvars>::compute_flux_interior(const fint iface,
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
		flux *= (-diffusivity*len);

		/// We assemble the negative of the residual r in 'M du/dt + r(u) = 0'
#pragma omp atomic
		residual(lelem,ivar) -= flux;

		if(relem < m->gnelem()) {
#pragma omp atomic
			residual(relem,ivar) += flux;
		}
	}
}

template<int nvars>
StatusCode DiffusionMA<nvars>::compute_residual(const Vec uvec, Vec rvec,
                                                const bool gettimesteps, Vec timesteps) const
{
	StatusCode ierr = 0;

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	const amat::Array2dView<freal> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

	const ConstGhostedVecHandler<freal> uvh(uvec);
	const freal *const uarr = uvh.getArray();
	Eigen::Map<const MVector<freal>> u(uarr, m->gnelem()+m->gnConnFace(), nvars);

	amat::Array2d<freal> uleft;
	amat::Array2d<freal> ug;
	uleft.resize(m->gnbface(),nvars);	// Modified
	ug.resize(m->gnbface(),nvars);

	freal *const ugptr = m->gnbface() > 0 ? &ug(0,0) : nullptr;

	for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const fint ielem = m->gintfac(ied,0);
		for(int ivar = 0; ivar < nvars; ivar++)
			uleft(ied - m->gPhyBFaceStart(),ivar) = u(ielem,ivar);
	}

	compute_boundary_states(&uleft(0,0), &ug(0,0));

	Vec gradvec;
	ierr = createGhostedSystemVector(m, NDIM*nvars, &gradvec); CHKERRQ(ierr);
	{
		MutableGhostedVecHandler<freal> grh(gradvec);
		freal *const gradarray = grh.getArray();
		const amat::Array2dView<freal> ua(uarr, m->gnelem()+m->gnConnFace(), nvars);

		gradcomp->compute_gradients(ua, amat::Array2dView<freal>(ugptr,m->gnbface(),nvars),
		                            gradarray);
	}
	//std::cout << "Computed gradients." << std::endl;

	ierr = VecGhostUpdateBegin(gradvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

	{
		MutableGhostedVecHandler<freal> rvh(rvec);
		freal *const rarr = rvh.getArray();
		amat::Array2dMutableView<freal> residual(rarr, m->gnelem()+m->gnConnFace(), nvars);

		ConstGhostedVecHandler<freal> grh(gradvec);
		const GradBlock_t<freal,NDIM,nvars> *const grads
			= reinterpret_cast<const GradBlock_t<freal,NDIM,nvars>*>(grh.getArray());

#pragma omp parallel for default(shared)
		for(int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
		{
			const fint lelem = m->gintfac(iface,0);
			const fint ibpface = iface - m->gPhyBFaceStart();
			const freal len = m->gfacemetric(iface,2);

			freal gradl[NDIM*nvars], gradr[NDIM*nvars];
			for(int ivar = 0; ivar < nvars; ivar++) {
				for(int idim = 0; idim < NDIM; idim++) {
					gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
					gradr[idim*nvars+ivar] = grads[lelem](idim,ivar);
				}
			}

			freal gradf[NDIM][nvars];
			getFaceGradient_modifiedAverage(&rc(lelem,0), &rcbp(ibpface,0),
			                                &uarr[lelem*nvars], &ug(ibpface,0), gradl, gradr, gradf);

			for(int ivar = 0; ivar < nvars; ivar++)
			{
				// compute nu*(-grad u . n) * l
				freal flux = 0;
				for(int idim = 0; idim < NDIM; idim++)
					flux += gradf[idim][ivar]*m->gfacemetric(iface,idim);
				flux *= (-diffusivity*len);

				/// NOTE: we assemble the negative of the residual r in 'M du/dt + r(u) = 0'
#pragma omp atomic
				residual(lelem,ivar) -= flux;
			}
		}
	}

	ierr = VecGhostUpdateEnd(gradvec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

	{
		MutableGhostedVecHandler<freal> rvh(rvec);
		freal *const rarr = rvh.getArray();
		amat::Array2dMutableView<freal> residual(rarr, m->gnelem()+m->gnConnFace(), nvars);

		ConstGhostedVecHandler<freal> grh(gradvec);
		const GradBlock_t<freal,NDIM,nvars> *const grads
			= reinterpret_cast<const GradBlock_t<freal,NDIM,nvars>*>(grh.getArray());

#pragma omp parallel for default(shared)
		for(fint iface = m->gDomFaceStart(); iface < m->gDomFaceEnd(); iface++)
		{
			compute_flux_interior(iface, rc, uarr, grads, residual);
		}
	}

	ierr = VecDestroy(&gradvec); CHKERRQ(ierr);

	{
		MutableVecHandler<freal> dtvh(timesteps);
		freal *const dtm = dtvh.getArray();
		MutableVecHandler<freal> rvh(rvec);
		freal *const rarr = rvh.getArray();
		Eigen::Map<MVector<freal>> residual(rarr, m->gnelem(), nvars);

#pragma omp parallel for default(shared)
		for(int iel = 0; iel < m->gnelem(); iel++)
		{
			if(gettimesteps)
				dtm[iel] = h[iel]*h[iel]/diffusivity;

			// subtract source term
			freal sourceterm[nvars];
			source(&rc(iel,0), 0, &uarr[iel*nvars], sourceterm);
			for(int ivar = 0; ivar < nvars; ivar++)
				residual(iel,ivar) += sourceterm[ivar]*m->garea(iel);
		}
	}

	return ierr;
}

template<int nvars>
void DiffusionMA<nvars>
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
void DiffusionMA<nvars>
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

template <int nvars>
void DiffusionMA<nvars>::getGradients(const Vec uvec,
                                      GradBlock_t<freal,NDIM,nvars> *const grads) const
{
	amat::Array2d<freal> ug(m->gnbface(),nvars);
	ConstGhostedVecHandler<freal> uh(uvec);
	const amat::Array2dView<freal> u(uh.getArray(), m->gnelem()+m->gnConnFace(), nvars);
	amat::Array2d<freal> uleft(m->gnbface(), nvars);
	for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const fint ielem = m->gintfac(ied,0);
		for(int ivar = 0; ivar < nvars; ivar++)
			uleft(ied - m->gPhyBFaceStart(),ivar) = u(ielem,ivar);
	}

	compute_boundary_states(&uleft(0,0), &ug(0,0));

	gradcomp->compute_gradients(u, amat::Array2dView<freal>(&ug(0,0),m->gnbface(),nvars),
	                            &grads[0](0,0));
}

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh<freal,NDIM> *const m, const Vec uvec,
                                    amat::Array2d<freal>& up)
{
	std::cout << "postprocess_point(): Creating output arrays\n";

	std::vector<freal> areasum(m->gnpoin(),0);
	up.resize(m->gnpoin(), nvars);
	up.zeros();
	//areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<freal>> u(uarr, m->gnelem(), NVARS);

	for(fint ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int inode = 0; inode < m->gnnode(ielem); inode++)
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				up(m->ginpoel(ielem,inode),ivar) += u(ielem,ivar)*m->garea(ielem);
				areasum[m->ginpoel(ielem,inode)] += m->garea(ielem);
			}
	}

	for(fint ipoin = 0; ipoin < m->gnpoin(); ipoin++)
		for(int ivar = 0; ivar < nvars; ivar++)
			up(ipoin,ivar) /= areasum[ipoin];

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	return ierr;
}

// template instantiations

//CHANGE HERE
template class Diffusion<1>;
template class DiffusionMA<1>;
template StatusCode scalar_postprocess_point<1>(const UMesh<freal,NDIM> *const m, const Vec uvec,
                                                amat::Array2d<freal>& up);
}

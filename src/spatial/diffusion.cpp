#include <iostream>
#include "diffusion.hpp"
#include "utilities/afactory.hpp"
#include "linalg/petscutils.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template<int nvars>
Diffusion<nvars>::Diffusion(const UMesh2dh<a_real> *const mesh,
                            const a_real diffcoeff, const a_real bvalue,
                            std::function <
								void(const a_real *const, const a_real, const a_real *const,
								     a_real *const)
                            > sourcefunc)
	: Spatial<a_real,nvars>(mesh), diffusivity{diffcoeff}, bval{bvalue}, source(sourcefunc)
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

// Currently, all boundaries are constant Dirichlet
template<int nvars>
inline void Diffusion<nvars>::compute_boundary_state(const int ied,
		const a_real *const ins, a_real *const bs) const
{
	for(int ivar = 0; ivar < nvars; ivar++)
		bs[ivar] = 2.0*bval - ins[ivar];
}

template<int nvars>
void Diffusion<nvars>::compute_boundary_states(const a_real *const instates,
                                               a_real *const bounstates) const
{
	for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		compute_boundary_state(ied,
		                       &instates [(ied-m->gPhyBFaceStart())*nvars],
		                       &bounstates[(ied-m->gPhyBFaceStart())*nvars]);
}

template<int nvars>
DiffusionMA<nvars>
::DiffusionMA(const UMesh2dh<a_real> *const mesh,
              const a_real diffcoeff, const a_real bvalue,
              std::function<void(const a_real *const,const a_real,const a_real *const,a_real *const)> sf,
              const std::string grad_scheme)
	: Diffusion<nvars>(mesh, diffcoeff, bvalue, sf),
	gradcomp {create_const_gradientscheme<a_real,nvars>(grad_scheme, m, &rc(0,0))}
{ }

template<int nvars>
DiffusionMA<nvars>::~DiffusionMA()
{
	delete gradcomp;
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

	const a_real *uarr = getVecAsReadOnlyArray<a_real>(uvec);
	Eigen::Map<const MVector<a_real>> u(uarr, m->gnelem(), nvars);

	a_real *rarr = getVecAsArray<a_real>(rvec);
	Eigen::Map<MVector<a_real>> residual(rarr, m->gnelem(), nvars);

	amat::Array2d<a_real> uleft;
	amat::Array2d<a_real> ug;
	uleft.resize(m->gnbface(),nvars);	// Modified
	ug.resize(m->gnbface(),nvars);

	for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		for(int ivar = 0; ivar < nvars; ivar++)
			uleft(ied - m->gPhyBFaceStart(),ivar) = u(ielem,ivar);
	}

	std::vector<GradBlock_t<a_real,NDIM,nvars>> grads;
	grads.resize(m->gnelem());

	compute_boundary_states(&uleft(0,0), &ug(0,0));
	gradcomp->compute_gradients(u, ug, &grads[0]);

#pragma omp parallel for default(shared)
	for(a_int iface = m->gDomFaceStart(); iface < m->gDomFaceEnd(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);
		const a_real len = m->gfacemetric(iface,2);

		a_real gradl[NDIM*nvars], gradr[NDIM*nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
				gradr[idim*nvars+ivar] = grads[relem](idim,ivar);
			}
		}
		// const a_real *const gradl = &grads[lelem](0,0);
		// const a_real *const gradr = &grads[relem](0,0);

		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage
			(&rc(lelem,0), &rc(relem,0), &uarr[lelem*nvars], &uarr[relem*nvars], gradl, gradr, gradf);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(-grad u . n) * l
			a_real flux = 0;
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

#pragma omp parallel for default(shared)
	for(int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);    // ghost cell
		const a_real len = m->gfacemetric(iface,2);

		a_real gradl[NDIM*nvars], gradr[NDIM*nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
				gradr[idim*nvars+ivar] = grads[lelem](idim,ivar);
			}
		}
		// const a_real *const gradl = &grads[lelem](0,0);
		// const a_real *const gradr = &grads[lelem](0,0);

		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage
			(&rc(lelem,0), &rc(relem,0), &uarr[lelem*nvars], &ug(iface-m->gPhyBFaceStart(),0),
			 gradl, gradr, gradf);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			// compute nu*(-grad u . n) * l
			a_real flux = 0;
			for(int idim = 0; idim < NDIM; idim++)
				flux += gradf[idim][ivar]*m->gfacemetric(iface,idim);
			flux *= (-diffusivity*len);

			/// NOTE: we assemble the negative of the residual r in 'M du/dt + r(u) = 0'
			residual(lelem,ivar) -= flux;
		}
	}

	a_real *dtm = nullptr;
	if(gettimesteps)
		dtm = getVecAsArray<a_real>(timesteps);

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

	if(gettimesteps)
		restoreArraytoVec(timesteps, &dtm);

	restoreArraytoVec<a_real>(rvec, &rarr);
	restoreReadOnlyArraytoVec<a_real>(uvec, &uarr);

	return ierr;
}

template<int nvars>
void DiffusionMA<nvars>
::compute_local_jacobian_interior(const a_int iface,
                                  const a_real *const ul, const a_real *const ur,
                                  Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& L,
                                  Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& U) const
{
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
	getFaceGradientAndJacobian_thinLayer(&rc(lelem), &rc(relem), ul, ur, du, du, grad, dgradl, dgradr);

	L = Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>::Zero();
	U = Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>::Zero();
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
::compute_local_jacobian_boundary(const a_int iface,
                                  const a_real *const ul,
                                  Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>& L) const
{
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

	// Compute the face gradient and its Jacobian; we don't actually need the gradient, however
	getFaceGradientAndJacobian_thinLayer(&rc(lelem), &rc(relem), ul, ul, du, du, grad, dgradl, dgradr);

	L = Eigen::Matrix<a_real,nvars,nvars,Eigen::RowMajor>::Zero();
	for(int ivar = 0; ivar < nvars; ivar++)
	{
		// compute nu*(d(-grad u)/du_l . n) * l
		for(int idim = 0; idim < NDIM; idim++)
			L(ivar,ivar) += dgradl[idim][ivar][ivar]*m->gfacemetric(iface,idim);
		L(ivar,ivar) *= (diffusivity*len);
	}
}

template <int nvars>
void DiffusionMA<nvars>::getGradients(const MVector<a_real>& u,
                                      GradBlock_t<a_real,NDIM,nvars> *const grads) const
{
	amat::Array2d<a_real> ug(m->gnbface(),nvars);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface+m->gPhyBFaceStart(),0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface,0));
	}

	gradcomp->compute_gradients(u, ug, &grads[0]);
}

template<int nvars>
StatusCode scalar_postprocess_point(const UMesh2dh<a_real> *const m, const Vec uvec,
                                    amat::Array2d<a_real>& up)
{
	std::cout << "Diffusion: postprocess_point(): Creating output arrays\n";

	std::vector<a_real> areasum(m->gnpoin(),0);
	up.resize(m->gnpoin(), nvars);
	up.zeros();
	//areasum.zeros();

	StatusCode ierr = 0;
	const PetscScalar* uarr;
	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);
	Eigen::Map<const MVector<a_real>> u(uarr, m->gnelem(), NVARS);

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

// template instantiations

//CHANGE HERE
template class Diffusion<1>;
template class DiffusionMA<1>;
template StatusCode scalar_postprocess_point<1>(const UMesh2dh<a_real> *const m, const Vec uvec,
                                                amat::Array2d<a_real>& up);
}

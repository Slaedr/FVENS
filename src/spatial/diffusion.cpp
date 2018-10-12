#include <iostream>
#include "diffusion.hpp"
#include "utilities/afactory.hpp"

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
DiffusionMA<nvars>::DiffusionMA(const UMesh2dh<a_real> *const mesh, 
		const a_real diffcoeff, const a_real bvalue,
	std::function<void(const a_real *const,const a_real,const a_real *const,a_real *const)> sf, 
		const std::string grad_scheme)
	: Diffusion<nvars>(mesh, diffcoeff, bvalue, sf),
	  gradcomp {create_const_gradientscheme<a_real,nvars>(grad_scheme, m, rc)}
{ }

template<int nvars>
DiffusionMA<nvars>::~DiffusionMA()
{
	delete gradcomp;
}

template<int nvars>
StatusCode DiffusionMA<nvars>::assemble_residual(const Vec uvec,
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
	ierr = VecGetArray(rvec, &rarr); CHKERRQ(ierr);

	ierr = compute_residual(uarr, rarr, gettimesteps, dtm); CHKERRQ(ierr);
	
	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);
	ierr = VecRestoreArray(rvec, &rarr); CHKERRQ(ierr);
	return ierr;
}

template<int nvars>
StatusCode DiffusionMA<nvars>::compute_residual(const a_real *const uarr,
                                                a_real *const __restrict rarr, 
                                                const bool gettimesteps, 
                                                std::vector<a_real>& dtm) const
{
	StatusCode ierr = 0;

	Eigen::Map<const MVector<a_real>> u(uarr, m->gnelem(), nvars);
	Eigen::Map<MVector<a_real>> residual(rarr, m->gnelem(), nvars);

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

	GradArray<a_real,nvars> grads;
	grads.resize(m->gnelem());
	
	compute_boundary_states(uleft, ug);
	gradcomp->compute_gradients(u, ug, grads);

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);
		const a_real len = m->gfacemetric(iface,2);
		
		/*a_real gradl[NDIM*nvars], gradr[NDIM*nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
				gradr[idim*nvars+ivar] = grads[relem](idim,ivar);
			}
			}*/
		const a_real *const gradl = &grads[lelem](0,0);
		const a_real *const gradr = &grads[relem](0,0);
	
		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage
			(iface, &uarr[lelem*nvars], &uarr[relem*nvars], gradl, gradr, gradf);

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
#pragma omp atomic
			residual(relem,ivar) += flux;
		}
	}
	
#pragma omp parallel for default(shared)
	for(int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_real len = m->gfacemetric(iface,2);
		
		/*a_real gradl[NDIM*nvars], gradr[NDIM*nvars];
		for(int ivar = 0; ivar < nvars; ivar++) {
			for(int idim = 0; idim < NDIM; idim++) {
				gradl[idim*nvars+ivar] = grads[lelem](idim,ivar);
				gradr[idim*nvars+ivar] = grads[lelem](idim,ivar);
			}
		}*/
		const a_real *const gradl = &grads[lelem](0,0);
		const a_real *const gradr = &grads[lelem](0,0);

		a_real gradf[NDIM][nvars];
		getFaceGradient_modifiedAverage
			(iface, &uarr[lelem*nvars], &ug(iface,0), gradl, gradr, gradf);

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
void DiffusionMA<nvars>::getGradients(const MVector<a_real>& u,
                                      GradArray<a_real,nvars>& grads) const
{
	amat::Array2d<a_real> ug(m->gnbface(),nvars);
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int lelem = m->gintfac(iface,0);
		compute_boundary_state(iface, &u(lelem,0), &ug(iface,0));
	}

	gradcomp->compute_gradients(u, ug, grads);
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

template class Diffusion<1>;
template class DiffusionMA<1>;
template StatusCode scalar_postprocess_point<1>(const UMesh2dh<a_real> *const m, const Vec uvec,
                                                amat::Array2d<a_real>& up);
}

/** @file agradientschemes.cpp
 * @brief Implementations of gradient estimation schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#include "agradientschemes.hpp"
#include <Eigen/LU>

namespace fvens
{

template<typename scalar, int nvars>
GradientScheme<scalar,nvars>::GradientScheme(const UMesh2dh<scalar> *const mesh, 
		const amat::Array2d<scalar>& _rc)
	: m{mesh}, rc{_rc}
{ }

template<typename scalar, int nvars>
GradientScheme<scalar,nvars>::~GradientScheme()
{ }

template<typename scalar, int nvars>
ZeroGradients<scalar,nvars>::ZeroGradients(const UMesh2dh<scalar> *const mesh, 
		const amat::Array2d<scalar>& _rc)
	: GradientScheme<scalar,nvars>(mesh, _rc)
{ }

template<typename scalar, int nvars>
void ZeroGradients<scalar,nvars>::compute_gradients(
		const MVector<scalar>& u, 
		const amat::Array2d<scalar>& ug, 
		GradArray<scalar,nvars>& grad ) const
{
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int j = 0; j < NDIM; j++)
			for(int i = 0; i < nvars; i++)
				grad[iel](j,i) = 0;
	}
}

template<typename scalar, int nvars>
GreenGaussGradients<scalar,nvars>::GreenGaussGradients(const UMesh2dh<scalar> *const mesh, 
		const amat::Array2d<scalar>& _rc)
	: GradientScheme<scalar,nvars>(mesh, _rc)
{ }

/* The state at the face is approximated as an inverse-distance-weighted average.
 */
template<typename scalar, int nvars>
void GreenGaussGradients<scalar,nvars>::compute_gradients(
		const MVector<scalar>& u, 
		const amat::Array2d<scalar>& ug, 
		GradArray<scalar,nvars>& grad ) const
{
#pragma omp parallel default(shared)
	{
#pragma omp for
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int j = 0; j < NDIM; j++)
				for(int i = 0; i < nvars; i++)
					grad[iel](j,i) = 0;
		}
		
#pragma omp for
		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			scalar areainv1;
			scalar ut[nvars];
			scalar dL, dR, mid[NDIM];
		
			const a_int ielem = m->gintfac(iface,0);
			const a_int jelem = m->gintfac(iface,1);   // ghost cell index
			const a_int ip1 = m->gintfac(iface,2);
			const a_int ip2 = m->gintfac(iface,3);
			dL = 0; dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				mid[idim] = (m->gcoords(ip1,idim) + m->gcoords(ip2,idim)) * 0.5;
				dL += (mid[idim]-rc(ielem,idim))*(mid[idim]-rc(ielem,idim));
				dR += (mid[idim]-rc(jelem,idim))*(mid[idim]-rc(jelem,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			areainv1 = 1.0/m->garea(ielem);
			
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				ut[ivar]= (u(ielem,ivar)*dL + ug(iface,ivar)*dR)/(dL+dR) * m->gfacemetric(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
					grad[ielem](idim,ivar) += (ut[ivar] * m->gfacemetric(iface,idim))*areainv1;
			}
		}

#pragma omp for
		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			a_int ielem, jelem, ip1, ip2;
			scalar areainv1, areainv2;
			scalar ut[nvars];
			scalar dL, dR, mid[NDIM];
		
			ielem = m->gintfac(iface,0);
			jelem = m->gintfac(iface,1);
			ip1 = m->gintfac(iface,2);
			ip2 = m->gintfac(iface,3);
			dL = 0; dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				mid[idim] = (m->gcoords(ip1,idim) + m->gcoords(ip2,idim)) * 0.5;
				dL += (mid[idim]-rc(ielem,idim))*(mid[idim]-rc(ielem,idim));
				dR += (mid[idim]-rc(jelem,idim))*(mid[idim]-rc(jelem,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			areainv1 = 1.0/m->garea(ielem);
			areainv2 = 1.0/m->garea(jelem);
			
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				ut[ivar] = (u(ielem,ivar)*dL + u(jelem,ivar)*dR)/(dL+dR) * m->gfacemetric(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
				{
#pragma omp atomic update
					grad[ielem](idim,ivar) += (ut[ivar] * m->gfacemetric(iface,idim))*areainv1;
#pragma omp atomic update
					grad[jelem](idim,ivar) -= (ut[ivar] * m->gfacemetric(iface,idim))*areainv2;
				}
			}
		}
	} // end parallel region
}

/** An inverse-distance weighted least-squares is used.
 */
template<typename scalar, int nvars>
WeightedLeastSquaresGradients<scalar,nvars>::WeightedLeastSquaresGradients(
		const UMesh2dh<scalar> *const mesh, 
		const amat::Array2d<scalar>& _rc)
	: GradientScheme<scalar,nvars>(mesh, _rc)
{ 
	V.resize(m->gnelem());
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		V[iel] = Matrix<scalar,NDIM,NDIM>::Zero();
	}

	// compute LHS of least-squares problem

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += (rc(ielem,idim)-rc(jelem,idim))*(rc(ielem,idim)-rc(jelem,idim));
			dr[idim] = rc(ielem,idim)-rc(jelem,idim);
		}
		w2 = 1.0/(w2);
		
		for(int i = 0; i<NDIM; i++)
			for(int j = 0; j < NDIM; j++) {
#pragma omp atomic update
				V[ielem](i,j) += w2*dr[i]*dr[j];
			}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rc(ielem,idim)-rc(jelem,idim))*(rc(ielem,idim)-rc(jelem,idim));
			dr[idim] = rc(ielem,idim)-rc(jelem,idim);
		}
		w2 = 1.0/(w2);
		
		for(int i = 0; i<NDIM; i++)
			for(int j = 0; j < NDIM; j++) {
#pragma omp atomic update
				V[ielem](i,j) += w2*dr[i]*dr[j];
#pragma omp atomic update
				V[jelem](i,j) += w2*dr[i]*dr[j];
			}
	}

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		V[ielem] = V[ielem].inverse().eval();
	}
}

template <typename scalar, int nvars>
using FMultiVectorArray = std::vector<Matrix<scalar,NDIM,nvars>,
                                      aligned_allocator<Matrix<scalar,NDIM,nvars>> >;

template<typename scalar, int nvars>
void WeightedLeastSquaresGradients<scalar,nvars>::compute_gradients(
		const MVector<scalar>& u, 
		const amat::Array2d<scalar>& ug, 
		GradArray<scalar,nvars>& grad ) const
{
	FMultiVectorArray<scalar,nvars> f;
	f.resize(m->gnelem());

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
		f[ielem] = Matrix<scalar,NDIM,nvars>::Zero();
	
	// compute least-squares RHS

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM], du[nvars];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += (rc(ielem,idim)-rc(jelem,idim))*(rc(ielem,idim)-rc(jelem,idim));
			dr[idim] = rc(ielem,idim)-rc(jelem,idim);
		}
		w2 = 1.0/(w2);
		
		for(short ivar = 0; ivar < nvars; ivar++)
			du[ivar] = u(ielem,ivar) - ug(iface,ivar);
		
		for(short ivar = 0; ivar < nvars; ivar++)
		{
			for(int jdim = 0; jdim < NDIM; jdim++)
#pragma omp atomic update
				f[ielem](jdim,ivar) += w2*dr[jdim]*du[ivar];
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM], du[nvars];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += (rc(ielem,idim)-rc(jelem,idim))*(rc(ielem,idim)-rc(jelem,idim));
			dr[idim] = rc(ielem,idim)-rc(jelem,idim);
		}
		w2 = 1.0/(w2);
		
		for(short ivar = 0; ivar < nvars; ivar++)
			du[ivar] = u(ielem,ivar) - u(jelem,ivar);

		for(short ivar = 0; ivar < nvars; ivar++)
		{
			for(int jdim = 0; jdim < NDIM; jdim++) {
#pragma omp atomic update
				f[ielem](jdim,ivar) += w2*dr[jdim]*du[ivar];
#pragma omp atomic update
				f[jelem](jdim,ivar) += w2*dr[jdim]*du[ivar];
			}
		}
	}

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		Matrix <scalar,NDIM,nvars> d = V[ielem]*f[ielem];
		for(short ivar = 0; ivar < nvars; ivar++)
		{
			for(short idim = 0; idim < NDIM; idim++)
				grad[ielem](idim,ivar) = d(idim,ivar);
		}
	}
}

template class ZeroGradients<a_real,NVARS>;
template class GreenGaussGradients<a_real,NVARS>;
template class WeightedLeastSquaresGradients<a_real,NVARS>;
template class ZeroGradients<a_real,1>;
template class GreenGaussGradients<a_real,1>;
template class WeightedLeastSquaresGradients<a_real,1>;

} // end namespace

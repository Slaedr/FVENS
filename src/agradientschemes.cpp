/** @file agradientschemes.cpp
 * @brief Implementations for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#include "agradientschemes.hpp"
#include <Eigen/LU>

namespace acfd
{

template<short nvars>
GradientScheme<nvars>::GradientScheme(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& _rc)
	: m{mesh}, rc{_rc}
{ }

template<short nvars>
GradientScheme<nvars>::~GradientScheme()
{ }

template<short nvars>
ZeroGradients<nvars>::ZeroGradients(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& _rc)
	: GradientScheme<nvars>(mesh, _rc)
{ }

template<short nvars>
void ZeroGradients<nvars>::compute_gradients(
		const MVector& u, 
		const amat::Array2d<a_real>& ug, 
		std::vector<FArray<NDIM,nvars>>& grad ) const
{
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int j = 0; j < NDIM; j++)
			for(int i = 0; i < nvars; i++)
				grad[iel](j,i) = 0;
	}
}

template<short nvars>
GreenGaussGradients<nvars>::GreenGaussGradients(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& _rc)
	: GradientScheme<nvars>(mesh, _rc)
{ }

/* The state at the face is approximated as an inverse-distance-weighted average.
 */
template<short nvars>
void GreenGaussGradients<nvars>::compute_gradients(
		const MVector& u, 
		const amat::Array2d<a_real>& ug, 
		std::vector<FArray<NDIM,nvars>>& grad ) const
{
#pragma omp parallel default(shared)
	{
#pragma omp for simd
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int j = 0; j < NDIM; j++)
				for(int i = 0; i < nvars; i++)
					grad[iel](j,i) = 0;
		}
		
#pragma omp for
		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			a_real areainv1;
			a_real ut[nvars];
			a_real dL, dR, mid[NDIM];
		
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
				ut[ivar]= (u(ielem,ivar)*dL + ug(iface,ivar)*dR)/(dL+dR) * m->ggallfa(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
					grad[ielem](idim,ivar) += (ut[ivar] * m->ggallfa(iface,idim))*areainv1;
			}
		}

#pragma omp for
		for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
		{
			a_int ielem, jelem, ip1, ip2;
			a_real areainv1, areainv2;
			a_real ut[nvars];
			a_real dL, dR, mid[NDIM];
		
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
				ut[ivar] = (u(ielem,ivar)*dL + u(jelem,ivar)*dR)/(dL+dR) * m->ggallfa(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
				{
#pragma omp atomic update
					grad[ielem](idim,ivar) += (ut[ivar] * m->ggallfa(iface,idim))*areainv1;
#pragma omp atomic update
					grad[jelem](idim,ivar) -= (ut[ivar] * m->ggallfa(iface,idim))*areainv2;
				}
			}
		}
	} // end parallel region
}

/** An inverse-distance weighted least-squares is used.
 */
template<short nvars>
WeightedLeastSquaresGradients<nvars>::WeightedLeastSquaresGradients(
		const UMesh2dh *const mesh, 
		const amat::Array2d<a_real>& _rc)
	: GradientScheme<nvars>(mesh, _rc)
{ 
	V.resize(m->gnelem());
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		V[iel] = Matrix<a_real,NDIM,NDIM>::Zero();
	}

	// compute LHS of least-squares problem

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		a_real w2 = 0, dr[NDIM];
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
		a_real w2 = 0, dr[NDIM];
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

template<short nvars>
void WeightedLeastSquaresGradients<nvars>::compute_gradients(
		const MVector& u, 
		const amat::Array2d<a_real>& ug, 
		std::vector<FArray<NDIM,nvars>>& grad ) const
{
	std::vector<Matrix<a_real,NDIM,nvars>, Eigen::aligned_allocator<Matrix<a_real,NDIM,nvars>> > f;
	f.resize(m->gnelem());

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
		f[ielem] = Matrix<a_real,NDIM,nvars>::Zero();
	
	// compute least-squares RHS

#pragma omp parallel for default(shared)
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		a_real w2 = 0, dr[NDIM], du[nvars];
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
		a_real w2 = 0, dr[NDIM], du[nvars];
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
		Matrix <a_real,NDIM,nvars> d = V[ielem]*f[ielem];
		for(short ivar = 0; ivar < nvars; ivar++)
		{
			for(short idim = 0; idim < NDIM; idim++)
				grad[ielem](idim,ivar) = d(idim,ivar);
		}
	}
}

template class ZeroGradients<NVARS>;
template class GreenGaussGradients<NVARS>;
template class WeightedLeastSquaresGradients<NVARS>;
template class ZeroGradients<1>;
template class GreenGaussGradients<1>;
template class WeightedLeastSquaresGradients<1>;

} // end namespace

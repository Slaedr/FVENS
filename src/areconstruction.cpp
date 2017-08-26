/** @file areconstruction.cpp
 * @brief Implementations for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#include "areconstruction.hpp"
#include <Eigen/LU>

namespace acfd
{

Reconstruction::Reconstruction(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real> *const _rc, const amat::Array2d<a_real>* const _rcg)
	: m(mesh), rc(_rc), rcg(_rcg)
{ }

Reconstruction::~Reconstruction()
{ }

template<short nvars>
ConstantReconstruction<nvars>::ConstantReconstruction(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real> *const _rc, const amat::Array2d<a_real>* const _rcg)
	: Reconstruction(mesh, _rc, _rcg)
{ }

template<short nvars>
void ConstantReconstruction<nvars>::compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const u, 
		const amat::Array2d<a_real>*const ug, 
		amat::Array2d<a_real>*const dudx, amat::Array2d<a_real>*const dudy)
{
#pragma omp parallel for simd default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		for(int i = 0; i < nvars; i++)
		{
			(*dudx)(iel,i) = 0;
			(*dudy)(iel,i) = 0;
		}
	}
}

template<short nvars>
GreenGaussReconstruction<nvars>::GreenGaussReconstruction(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real> *const _rc, const amat::Array2d<a_real>* const _rcg)
	: Reconstruction(mesh, _rc, _rcg)
{ }

/* The state at the face is approximated as an inverse-distance-weighted average.
 */
template<short nvars>
void GreenGaussReconstruction<nvars>::compute_gradients(const Matrix<a_real,Dynamic,Dynamic,RowMajor>*const u, 
		const amat::Array2d<a_real>*const ug, 
		amat::Array2d<a_real>*const dudx, amat::Array2d<a_real>*const dudy)
{
#pragma omp parallel default(shared)
	{
#pragma omp for simd
		for(a_int iel = 0; iel < m->gnelem(); iel++)
		{
			for(int i = 0; i < nvars; i++)
			{
				(*dudx)(iel,i) = 0;
				(*dudy)(iel,i) = 0;
			}
		}
		
#pragma omp for
		for(a_int iface = 0; iface < m->gnbface(); iface++)
		{
			a_int ielem, ip1, ip2;
			a_real areainv1;
			a_real ut[nvars];
			a_real dL, dR, mid[NDIM];
		
			ielem = m->gintfac(iface,0);
			ip1 = m->gintfac(iface,2);
			ip2 = m->gintfac(iface,3);
			dL = 0; dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				mid[idim] = (m->gcoords(ip1,idim) + m->gcoords(ip2,idim)) * 0.5;
				dL += (mid[idim]-(*rc)(ielem,idim))*(mid[idim]-(*rc)(ielem,idim));
				dR += (mid[idim]-(*rcg)(iface,idim))*(mid[idim]-(*rcg)(iface,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			areainv1 = 1.0/m->garea(ielem);
			
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				ut[ivar] = ((*u)(ielem,ivar)*dL + (*ug)(iface,ivar)*dR)/(dL+dR) * m->ggallfa(iface,2);
				(*dudx)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,0))*areainv1;
				(*dudy)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,1))*areainv1;
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
				dL += (mid[idim]-(*rc)(ielem,idim))*(mid[idim]-(*rc)(ielem,idim));
				dR += (mid[idim]-(*rc)(jelem,idim))*(mid[idim]-(*rc)(jelem,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			areainv1 = 1.0/m->garea(ielem);
			areainv2 = 1.0/m->garea(jelem);
			
			for(int ivar = 0; ivar < nvars; ivar++)
			{
				ut[ivar] = ((*u)(ielem,ivar)*dL + (*u)(jelem,ivar)*dR)/(dL+dR) * m->ggallfa(iface,2);
#pragma omp atomic update
				(*dudx)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,0))*areainv1;
#pragma omp atomic update
				(*dudy)(ielem,ivar) += (ut[ivar] * m->ggallfa(iface,1))*areainv1;
#pragma omp atomic update
				(*dudx)(jelem,ivar) -= (ut[ivar] * m->ggallfa(iface,0))*areainv2;
#pragma omp atomic update
				(*dudy)(jelem,ivar) -= (ut[ivar] * m->ggallfa(iface,1))*areainv2;
			}
		}
	} // end parallel region
}

/** An inverse-distance weighted least-squares is used.
 */
template<short nvars>
WeightedLeastSquaresReconstruction<nvars>::WeightedLeastSquaresReconstruction(const UMesh2dh *const mesh, 
		const amat::Array2d<a_real> *const _rc, const amat::Array2d<a_real>* const _rcg)
	: Reconstruction(mesh, _rc, _rcg)
{ 
	std::cout << "   WeightedLeastSquaresReconstruction: Setting up; num vars = " << nvars << std::endl;

	V.resize(m->gnelem());
	f.resize(m->gnelem());
	for(a_int i = 0; i < m->gnelem(); i++)
	{
		V[i] = Matrix<a_real,2,2>::Zero();
	}

	// compute LHS of least-squares problem

	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_real w2 = 0, dr[2];
		for(short idim = 0; idim < 2; idim++)
		{
			w2 += ((*rc)(ielem,idim)-(*rcg)(iface,idim))*((*rc)(ielem,idim)-(*rcg)(iface,idim));
			dr[idim] = (*rc)(ielem,idim)-(*rcg)(iface,idim);
		}
		w2 = 1.0/w2;

		V[ielem](0,0) += w2*dr[0]*dr[0];
		V[ielem](1,1) += w2*dr[1]*dr[1];
		V[ielem](0,1) += w2*dr[0]*dr[1];
		V[ielem](1,0) += w2*dr[0]*dr[1];
	}
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_int jelem = m->gintfac(iface,1);
		a_real w2 = 0, dr[2];
		for(short idim = 0; idim < 2; idim++)
		{
			w2 += ((*rc)(ielem,idim)-(*rc)(jelem,idim))*((*rc)(ielem,idim)-(*rc)(jelem,idim));
			dr[idim] = (*rc)(ielem,idim)-(*rc)(jelem,idim);
		}
		w2 = 1.0/w2;

		V[ielem](0,0) += w2*dr[0]*dr[0];
		V[ielem](1,1) += w2*dr[1]*dr[1];
		V[ielem](0,1) += w2*dr[0]*dr[1];
		V[ielem](1,0) += w2*dr[0]*dr[1];
		
		V[jelem](0,0) += w2*dr[0]*dr[0];
		V[jelem](1,1) += w2*dr[1]*dr[1];
		V[jelem](0,1) += w2*dr[0]*dr[1];
		V[jelem](1,0) += w2*dr[0]*dr[1];
	}

	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		//a_real det = V[ielem].determinant();
		//if(det < ZERO_TOL*100) std::cout << "  !!!! ERROR!\n";
		
		V[ielem] = V[ielem].inverse().eval();
		f[ielem] = Matrix<a_real,2,nvars>::Zero();
	}
}

template<short nvars>
void WeightedLeastSquaresReconstruction<nvars>::compute_gradients(
		const Matrix<a_real,Dynamic,Dynamic,RowMajor> *const u, 
		const amat::Array2d<a_real> *const ug, 
		amat::Array2d<a_real>*const dudx, amat::Array2d<a_real>*const dudy)
{
	// compute least-squares RHS

	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_real w2 = 0, dr[NDIM], du[nvars];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += ((*rc)(ielem,idim)-(*rcg)(iface,idim))*((*rc)(ielem,idim)-(*rcg)(iface,idim));
			dr[idim] = (*rc)(ielem,idim)-(*rcg)(iface,idim);
		}
		w2 = 1.0/w2;
		
		for(short ivar = 0; ivar < nvars; ivar++)
			du[ivar] = (*u)(ielem,ivar) - (*ug)(iface,ivar);
		
		for(short ivar = 0; ivar < nvars; ivar++)
		{
			f[ielem](0,ivar) += w2*dr[0]*du[ivar];
			f[ielem](1,ivar) += w2*dr[1]*du[ivar];
		}
	}
	for(a_int iface = m->gnbface(); iface < m->gnaface(); iface++)
	{
		a_int ielem = m->gintfac(iface,0);
		a_int jelem = m->gintfac(iface,1);
		a_real w2 = 0, dr[NDIM], du[nvars];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += ((*rc)(ielem,idim)-(*rc)(jelem,idim))*((*rc)(ielem,idim)-(*rc)(jelem,idim));
			dr[idim] = (*rc)(ielem,idim)-(*rc)(jelem,idim);
		}
		w2 = 1.0/w2;
		
		for(short ivar = 0; ivar < nvars; ivar++)
			du[ivar] = (*u)(ielem,ivar) - (*u)(jelem,ivar);

		for(short ivar = 0; ivar < nvars; ivar++)
		{
			f[ielem](0,ivar) += w2*dr[0]*du[ivar];
			f[ielem](1,ivar) += w2*dr[1]*du[ivar];
			f[jelem](0,ivar) += w2*dr[0]*du[ivar];
			f[jelem](1,ivar) += w2*dr[1]*du[ivar];
		}
	}

	// solve normal equations by Cramer's rule
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		d = V[ielem]*f[ielem];
		for(short ivar = 0; ivar < nvars; ivar++)
		{
			(*dudx)(ielem,ivar) = d(0,ivar);
			(*dudy)(ielem,ivar) = d(1,ivar);
		}
		f[ielem] = Matrix<a_real,2,nvars>::Zero();
	}
}

template class ConstantReconstruction<NVARS>;
template class GreenGaussReconstruction<NVARS>;
template class WeightedLeastSquaresReconstruction<NVARS>;
template class ConstantReconstruction<1>;
template class GreenGaussReconstruction<1>;
template class WeightedLeastSquaresReconstruction<1>;

} // end namespace

/** @file agradientschemes.cpp
 * @brief Implementations of gradient estimation schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

#include "agradientschemes.hpp"
#include <Eigen/LU>
#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif
#include "utilities/mpiutils.hpp"

namespace fvens
{

template<typename scalar, int nvars>
GradientScheme<scalar,nvars>::GradientScheme(const UMesh2dh<scalar> *const mesh,
                                             const scalar *const _rc,
                                             const scalar *const _rcbp)
	: m{mesh}, rc{_rc}, rcbp{_rcbp}
{
}

template<typename scalar, int nvars>
GradientScheme<scalar,nvars>::~GradientScheme()
{ }

template<typename scalar, int nvars>
ZeroGradients<scalar,nvars>::ZeroGradients(const UMesh2dh<scalar> *const mesh,
                                           const scalar *const _rc,
                                           const scalar *const _rcbp)
	: GradientScheme<scalar,nvars>(mesh, _rc, _rcbp)
{ }

template<typename scalar, int nvars>
void ZeroGradients<scalar,nvars>::compute_gradients(const amat::Array2dView<scalar> u,
                                                    const amat::Array2dView<scalar> ug,
                                                    scalar *const gradarray) const
{
	GradBlock_t<scalar,NDIM,nvars> *const grad
		= reinterpret_cast<GradBlock_t<scalar,NDIM,nvars>*>(gradarray);
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
                                                       const scalar *const _rc,
                                                       const scalar *const _rcbp)
	: GradientScheme<scalar,nvars>(mesh, _rc, _rcbp)
{ }

/* The state at the face is approximated as an inverse-distance-weighted average.
 */
template<typename scalar, int nvars>
void GreenGaussGradients<scalar,nvars>::compute_gradients(
		const amat::Array2dView<scalar> u,
		const amat::Array2dView<scalar> ug,
		scalar *const gradarray) const
{
	Eigen::Map<const MVector<scalar>> rcm(rc, m->gnelem()+m->gnConnFace(), NDIM);
	Eigen::Map<const MVector<scalar>> rcbpm(rcbp, m->gnbface(), NDIM);
	GradBlock_t<scalar,NDIM,nvars> *const grad
		= reinterpret_cast<GradBlock_t<scalar,NDIM,nvars>*>(gradarray);

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
		for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
		{
			const a_int ibpface = iface - m->gPhyBFaceStart();
			const a_int ielem = m->gintfac(iface,0);

			scalar mid[NDIM];
			for(int i = 0; i < NDIM; i++) mid[i] = 0;

			for(FIndex inofa = 2; inofa < m->gnnofa(iface)+2; inofa++)
			{
				const a_int ip = m->gintfac(iface,inofa);
				for(int idim = 0; idim < NDIM; idim++)
					mid[idim] += m->gcoords(ip,idim);
			}

			for(int i = 0; i < NDIM; i++) mid[i] /= m->gnnofa(iface);

			scalar dL = 0, dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				dL += (mid[idim]-rcm(ielem,idim))*(mid[idim]-rcm(ielem,idim));
				dR += (mid[idim]-rcbpm(ibpface,idim))*(mid[idim]-rcbpm(ibpface,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			const scalar areainv1 = 1.0/m->garea(ielem);

			for(int ivar = 0; ivar < nvars; ivar++)
			{
				const scalar ut = (u(ielem,ivar)*dL + ug(iface-m->gPhyBFaceStart(),ivar)*dR)/(dL+dR)
					* m->gfacemetric(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
					grad[ielem](idim,ivar) += (ut * m->gfacemetric(iface,idim))*areainv1;
			}
		}

#pragma omp for
		for(a_int iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++)
		{
			scalar dL, dR;

			const a_int ielem = m->gintfac(iface,0);
			const a_int jelem = m->gintfac(iface,1);

			scalar mid[NDIM];
			for(int i = 0; i < NDIM; i++) mid[i] = 0;

			for(FIndex inofa = 2; inofa < m->gnnofa(iface)+2; inofa++)
			{
				const a_int ip = m->gintfac(iface,inofa);
				for(int idim = 0; idim < NDIM; idim++)
					mid[idim] += m->gcoords(ip,idim);
			}

			for(int i = 0; i < NDIM; i++) mid[i] /= m->gnnofa(iface);

			dL = 0; dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				dL += (mid[idim]-rcm(ielem,idim))*(mid[idim]-rcm(ielem,idim));
				dR += (mid[idim]-rcm(jelem,idim))*(mid[idim]-rcm(jelem,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			const scalar areainv1 = 1.0/m->garea(ielem);
			const scalar areainv2 = 1.0/m->garea(jelem);

			for(int ivar = 0; ivar < nvars; ivar++)
			{
				const scalar ut = (u(ielem,ivar)*dL + u(jelem,ivar)*dR)/(dL+dR) * m->gfacemetric(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
				{
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
					grad[ielem](idim,ivar) += (ut * m->gfacemetric(iface,idim))*areainv1;
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
					grad[jelem](idim,ivar) -= (ut * m->gfacemetric(iface,idim))*areainv2;
				}
			}
		}

#pragma omp for
		for(a_int iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++)
		{
			scalar dL, dR;

			const a_int ielem = m->gintfac(iface,0);
			const a_int jelem = m->gintfac(iface,1);

			scalar mid[NDIM];
			for(int i = 0; i < NDIM; i++) mid[i] = 0;

			for(FIndex inofa = 2; inofa < m->gnnofa(iface)+2; inofa++)
			{
				const a_int ip = m->gintfac(iface,inofa);
				for(int idim = 0; idim < NDIM; idim++)
					mid[idim] += m->gcoords(ip,idim);
			}

			for(int i = 0; i < NDIM; i++) mid[i] /= m->gnnofa(iface);

			dL = 0; dR = 0;
			for(int idim = 0; idim < NDIM; idim++)
			{
				dL += (mid[idim]-rcm(ielem,idim))*(mid[idim]-rcm(ielem,idim));
				dR += (mid[idim]-rcm(jelem,idim))*(mid[idim]-rcm(jelem,idim));
			}
			dL = 1.0/sqrt(dL);
			dR = 1.0/sqrt(dR);
			const scalar areainv1 = 1.0/m->garea(ielem);

			for(int ivar = 0; ivar < nvars; ivar++)
			{
				const scalar ut = (u(ielem,ivar)*dL + u(jelem,ivar)*dR)/(dL+dR) * m->gfacemetric(iface,2);

				for(int idim = 0; idim < NDIM; idim++)
				{
					grad[ielem](idim,ivar) += (ut * m->gfacemetric(iface,idim))*areainv1;
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
		const scalar *const _rc,
		const scalar *const _rcbp)
	: GradientScheme<scalar,nvars>(mesh, _rc, _rcbp)
{
	Eigen::Map<const MVector<scalar>> rcm(rc, m->gnelem()+m->gnConnFace(), NDIM);
	Eigen::Map<const MVector<scalar>> rcbpm(rcbp, m->gnbface(), NDIM);

	V.resize(m->gnelem());
#pragma omp parallel for default(shared)
	for(a_int iel = 0; iel < m->gnelem(); iel++)
	{
		V[iel] = Eigen::Matrix<scalar,NDIM,NDIM>::Zero();
	}

	// compute LHS of least-squares problem

#pragma omp parallel for default(shared)
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int ibpface = iface - m->gPhyBFaceStart();
		const a_int ielem = m->gintfac(iface,0);

		scalar w2 = 0, dr[NDIM];
		for(short idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcbpm(ibpface,idim))*(rcm(ielem,idim)-rcbpm(ibpface,idim));
			dr[idim] = rcm(ielem,idim)-rcbpm(ibpface,idim);
		}
		w2 = 1.0/(w2);

		for(int i = 0; i<NDIM; i++)
			for(int j = 0; j < NDIM; j++) {
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				V[ielem](i,j) += w2*dr[i]*dr[j];
			}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcm(jelem,idim))*(rcm(ielem,idim)-rcm(jelem,idim));
			dr[idim] = rcm(ielem,idim)-rcm(jelem,idim);
		}
		w2 = 1.0/(w2);

		for(int i = 0; i<NDIM; i++)
			for(int j = 0; j < NDIM; j++) {
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				V[ielem](i,j) += w2*dr[i]*dr[j];
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				V[jelem](i,j) += w2*dr[i]*dr[j];
			}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcm(jelem,idim))*(rcm(ielem,idim)-rcm(jelem,idim));
			dr[idim] = rcm(ielem,idim)-rcm(jelem,idim);
		}
		w2 = 1.0/(w2);

		for(int i = 0; i<NDIM; i++)
			for(int j = 0; j < NDIM; j++) {
#pragma omp atomic update
				V[ielem](i,j) += w2*dr[i]*dr[j];
			}
	}

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		V[ielem] = V[ielem].inverse().eval();
	}
}

template <typename scalar, int nvars>
using FMultiVectorArray = std::vector<Eigen::Matrix<scalar,NDIM,nvars,Eigen::DontAlign>>;

template<typename scalar, int nvars>
void WeightedLeastSquaresGradients<scalar,nvars>::compute_gradients(const amat::Array2dView<scalar> u,
                                                                    const amat::Array2dView<scalar> ug,
                                                                    scalar *const gradarray) const
{
	Eigen::Map<const MVector<scalar>> rcm(rc, m->gnelem()+m->gnConnFace()+m->gnbface(), NDIM);
	Eigen::Map<const MVector<scalar>> rcbpm(rcbp, m->gnbface(), NDIM);
	GradBlock_t<scalar,NDIM,nvars> *const grad
		= reinterpret_cast<GradBlock_t<scalar,NDIM,nvars>*>(gradarray);
	FMultiVectorArray<scalar,nvars> f;
	f.resize(m->gnelem());

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
		f[ielem] = Eigen::Matrix<scalar,NDIM,nvars>::Zero();

	// compute least-squares RHS

#pragma omp parallel for default(shared)
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int ibpface = iface - m->gPhyBFaceStart();
		const a_int ielem = m->gintfac(iface,0);

		scalar w2 = 0, dr[NDIM], du[nvars];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcbpm(ibpface,idim))*(rcm(ielem,idim)-rcbpm(ibpface,idim));
			dr[idim] = rcm(ielem,idim)-rcbpm(ibpface,idim);
		}
		w2 = 1.0/(w2);

		for(int ivar = 0; ivar < nvars; ivar++)
			du[ivar] = u(ielem,ivar) - ug(iface-m->gPhyBFaceStart(),ivar);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			for(int jdim = 0; jdim < NDIM; jdim++)
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				f[ielem](jdim,ivar) += w2*dr[jdim]*du[ivar];
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM], du[nvars];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcm(jelem,idim))*(rcm(ielem,idim)-rcm(jelem,idim));
			dr[idim] = rcm(ielem,idim)-rcm(jelem,idim);
		}
		w2 = 1.0/(w2);

		for(int ivar = 0; ivar < nvars; ivar++)
			du[ivar] = u(ielem,ivar) - u(jelem,ivar);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			for(int jdim = 0; jdim < NDIM; jdim++) {
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				f[ielem](jdim,ivar) += w2*dr[jdim]*du[ivar];
#ifdef USE_ADOLC
#pragma omp critical
#else
#pragma omp atomic update
#endif
				f[jelem](jdim,ivar) += w2*dr[jdim]*du[ivar];
			}
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);
		const a_int jelem = m->gintfac(iface,1);
		scalar w2 = 0, dr[NDIM], du[nvars];
		for(int idim = 0; idim < NDIM; idim++)
		{
			w2 += (rcm(ielem,idim)-rcm(jelem,idim))*(rcm(ielem,idim)-rcm(jelem,idim));
			dr[idim] = rcm(ielem,idim)-rcm(jelem,idim);
		}
		w2 = 1.0/(w2);

		for(int ivar = 0; ivar < nvars; ivar++)
			du[ivar] = u(ielem,ivar) - u(jelem,ivar);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			for(int jdim = 0; jdim < NDIM; jdim++) {
#pragma omp atomic update
				f[ielem](jdim,ivar) += w2*dr[jdim]*du[ivar];
			}
		}
	}

#pragma omp parallel for default(shared)
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		/// \todo TODO: Optimize weighted least-squares final assignment
		Eigen::Matrix <scalar,NDIM,nvars> d = V[ielem]*f[ielem];
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

#ifdef USE_ADOLC
template class ZeroGradients<adouble,NVARS>;
template class GreenGaussGradients<adouble,NVARS>;
template class WeightedLeastSquaresGradients<adouble,NVARS>;
template class ZeroGradients<adouble,1>;
template class GreenGaussGradients<adouble,1>;
template class WeightedLeastSquaresGradients<adouble,1>;
#endif

} // end namespace

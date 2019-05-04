/** \file
 * \brief Implementation of some limited linear reconstruction schemes
 * \author Aditya Kashi
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
#include "mathutils.hpp"
#include "limitedlinearreconstruction.hpp"
#include "reconstruction_utils.hpp"

namespace fvens {

template <typename scalar, int nvars>
WENOReconstruction<scalar,nvars>::WENOReconstruction(const UMesh<scalar,2> *const mesh,
                                                     const scalar *const c_centres, 
                                                     const scalar *const c_centres_ghost,
                                                     const amat::Array2d<scalar>& gauss_r,
                                                     const freal l)
	: SolutionReconstruction<scalar,nvars>(mesh, c_centres, c_centres_ghost, gauss_r),
	  gamma{4.0}, lambda{l}, epsilon{1.0e-5}
{
}

template <typename scalar, int nvars>
void WENOReconstruction<scalar,nvars>
::compute_face_values(const MVector<scalar>& u, const amat::Array2dView<scalar> ug,
                      const scalar *const gradarray,
                      amat::Array2dMutableView<scalar> ufl,
                      amat::Array2dMutableView<scalar> ufr) const
{
	const GradBlock_t<scalar,NDIM,nvars> *const grads
		= reinterpret_cast<const GradBlock_t<scalar,NDIM,nvars>*>(gradarray);

	// first compute limited derivatives at each cell

#pragma omp parallel for default(shared)
	for(fint ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int ivar = 0; ivar < nvars; ivar++)
		{
			scalar wsum = 0;
			scalar lgrad[NDIM]; 
			zeros(lgrad, NDIM);

			// Central stencil
			{
				const scalar denom = pow( gradientMagnitude2(grads[ielem],ivar) + epsilon , gamma );
				const scalar w = lambda / denom;
				wsum += w;
				for(int j = 0; j < NDIM; j++)
					lgrad[j] += w*grads[ielem](j,ivar);
			}

			// Biased stencils
			for(int jel = 0; jel < m->gnfael(ielem); jel++)
			{
				const fint jelem = m->gesuel(ielem,jel);

				// ignore ghost cells
				if(jelem >= m->gnelem()+m->gnConnFace())
					continue;

				const scalar denom = pow( gradientMagnitude2(grads[jelem],ivar) + epsilon , gamma );
				const scalar w = 1.0 / denom;
				wsum += w;
				for(int j = 0; j < NDIM; j++)
					lgrad[j] += w*grads[jelem](j,ivar);
			}

			for(int j = 0; j < NDIM; j++)
				lgrad[j] /= wsum;
			
			for(int jfa = 0; jfa < m->gnfael(ielem); jfa++)
			{
				const fint face = m->gelemface(ielem,jfa);
				const fint jelem = m->gesuel(ielem,jfa);
				
				if(ielem < jelem) {
					ufl(face,ivar) = u(ielem,ivar);
					for(int j = 0; j < NDIM; j++)
						ufl(face,ivar) += lgrad[j]*(gr(face,j) - ri[ielem*NDIM+j]);
				}
				else {
					ufr(face,ivar) = u(ielem,ivar);
					for(int j = 0; j < NDIM; j++)
						ufr(face,ivar) += lgrad[j]*(gr(face,j) - ri[ielem*NDIM+j]);
				}
			}
		}
	}
}

template <typename scalar, int nvars>
BarthJespersenLimiter<scalar,nvars>::BarthJespersenLimiter(const UMesh<scalar,2> *const mesh, 
                                                           const scalar *const r_centres, 
                                                           const scalar *const r_centres_ghost,
                                                     const amat::Array2d<scalar>& gauss_r)
	: SolutionReconstruction<scalar,nvars>(mesh, r_centres, r_centres_ghost, gauss_r)
{
}

template <typename scalar, int nvars>
void BarthJespersenLimiter<scalar,nvars>::compute_face_values(const MVector<scalar>& u, 
                                                              const amat::Array2dView<scalar> ug, 
                                                              const scalar *const gradarray,
                                                              amat::Array2dMutableView<scalar> ufl,
                                                              amat::Array2dMutableView<scalar> ufr) const
{
	const GradBlock_t<scalar,NDIM,nvars> *const grads
		= reinterpret_cast<const GradBlock_t<scalar,NDIM,nvars>*>(gradarray);

#pragma omp parallel for default(shared)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ivar = 0; ivar < nvars; ivar++)
		{
			scalar duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint jel = m->gesuel(iel,j);
				const scalar dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			scalar lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint face = m->gelemface(iel,j);
				
				const scalar uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr(face,0), ri+iel*NDIM);
				
				scalar phiik;
				const scalar diff = uface - u(iel,ivar);
				if(diff>0)
					phiik = 1 < duimax/diff ? 1 : duimax/diff;
				else if(diff < 0)
					phiik = 1 < duimin/diff ? 1 : duimin/diff;
				else
					phiik = 1;

				if(phiik < lim)
					lim = phiik;
			}
			
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint face = m->gelemface(iel,j);
				const fint jel = m->gesuel(iel,j);
				
				if(iel < jel)
					ufl(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr(face,0), ri+iel*NDIM);
				else
					ufr(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr(face,0), ri+iel*NDIM);
			}

		}
	}
}

template <typename scalar, int nvars>
VenkatakrishnanLimiter<scalar,nvars>
::VenkatakrishnanLimiter(const UMesh<scalar,2> *const mesh,
                         const scalar *const r_centres, 
                         const scalar *const r_centres_ghost,
                         const amat::Array2d<scalar>& gauss_r,
                         const freal k_param)
	: SolutionReconstruction<scalar,nvars>(mesh, r_centres, r_centres_ghost, gauss_r), K{k_param}
{
	std::cout << "  Venkatakrishnan Limiter: Constant K = " << K << std::endl;
	// compute characteristic length, currently the maximum edge length, of all cells
	clength.resize(m->gnelem());
	static_assert(NDIM == 2, "Works only in 2D for now");
#pragma omp parallel for default(shared)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		for(int ifa = 0; ifa < m->gnnode(iel); ifa++)
		{
			scalar llen = 0;
			const int inode = ifa, jnode = (ifa+1) % m->gnnode(iel);
			for(int idim = 0; idim < NDIM; idim++)
				llen += std::pow(m->gcoords(m->ginpoel(iel,inode),idim) 
						- m->gcoords(m->ginpoel(iel,jnode),idim), 2);

			if(clength[iel] < llen) clength[iel] = llen;
		}
		clength[iel] = std::sqrt(clength[iel]);
	}
}

template <typename scalar, int nvars>
void VenkatakrishnanLimiter<scalar,nvars>
::compute_face_values(const MVector<scalar>& u,
                      const amat::Array2dView<scalar> ug, 
                      const scalar *const gradarray,
                      amat::Array2dMutableView<scalar> ufl,
                      amat::Array2dMutableView<scalar> ufr) const
{
	const GradBlock_t<scalar,NDIM,nvars> *const grads
		= reinterpret_cast<const GradBlock_t<scalar,NDIM,nvars>*>(gradarray);

#pragma omp parallel for default(shared)
	for(fint iel = 0; iel < m->gnelem(); iel++)
	{
		const scalar eps2 = std::pow(K*clength[iel], 3);

		for(int ivar = 0; ivar < nvars; ivar++)
		{
			scalar duimin=0, duimax=0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint jel = m->gesuel(iel,j);
				const scalar dui = u(jel,ivar)-u(iel,ivar);
				if(dui > duimax) duimax = dui;
				if(dui < duimin) duimin = dui;
			}
			
			scalar lim = 1.0;
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint face = m->gelemface(iel,j);
				
				const scalar uface = linearExtrapolate(u(iel,ivar), grads[iel], ivar, 1.0,
						&gr(face,0), ri+iel*NDIM);
				
				const scalar dm = uface - u(iel,ivar);

				// Venkatakrishnan modification
				const scalar dp = dm < 0 ? duimin : duimax;
				const scalar phiik = (dp*dp + 2*dp*dm + eps2)/(dp*dp + dp*dm + 2*dm*dm + eps2);

				if(phiik < lim)
					lim = phiik;
			}
			
			for(int j = 0; j < m->gnfael(iel); j++)
			{
				const fint face = m->gelemface(iel,j);
				const fint jel = m->gesuel(iel,j);
				
				if(iel < jel)
					ufl(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr(face,0), ri+iel*NDIM);
				else
					ufr(face,ivar) = linearExtrapolate(u(iel,ivar), grads[iel], ivar, lim,
						&gr(face,0), ri+iel*NDIM);
			}

		}
	}
}

template class WENOReconstruction<freal,NVARS>;
template class BarthJespersenLimiter<freal,NVARS>;
template class VenkatakrishnanLimiter<freal,NVARS>;
template class WENOReconstruction<freal,1>;
template class BarthJespersenLimiter<freal,1>;
template class VenkatakrishnanLimiter<freal,1>;

}

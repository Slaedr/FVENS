/** \file areconstruction.cpp
 * \brief Implementation of solution reconstruction schemes (limiters)
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
#include "areconstruction.hpp"
#include "reconstruction_utils.hpp"
#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template <typename scalar, int nvars>
SolutionReconstruction<scalar,nvars>::SolutionReconstruction (const UMesh<scalar,2> *const mesh,
                                                              const scalar *const c_centres,
                                                              const scalar *const c_centres_ghost,
                                                              const amat::Array2d<scalar>& gauss_r)
	: m{mesh}, ri{c_centres}, ribp{c_centres_ghost}, gr{gauss_r}
{ }

template <typename scalar, int nvars>
SolutionReconstruction<scalar,nvars>::~SolutionReconstruction()
{ }

template <typename scalar, int nvars>
LinearUnlimitedReconstruction<scalar,nvars>
::LinearUnlimitedReconstruction(const UMesh<scalar,2> *const mesh,
                                const scalar *const c_centres,
                                const scalar *const c_centres_ghost,
                                const amat::Array2d<scalar>& gauss_r)
	: SolutionReconstruction<scalar,nvars>(mesh, c_centres, c_centres_ghost, gauss_r)
{ }

template <typename scalar, int nvars>
void LinearUnlimitedReconstruction<scalar,nvars>
::compute_face_values(const MVector<scalar>& u,
                      const amat::Array2dView<scalar> ug,
                      const scalar *const gradarray,
                      amat::Array2dMutableView<scalar> ufl,
                      amat::Array2dMutableView<scalar> ufr) const
{
	const GradBlock_t<scalar,NDIM,nvars> *const grads
		= reinterpret_cast<const GradBlock_t<scalar,NDIM,nvars>*>(gradarray);

#pragma omp parallel default(shared)
	{
#pragma omp for nowait
		for(fint ied = m->gConnBFaceStart(); ied < m->gConnBFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr(ied,0), ri+ielem*NDIM);
			}
		}

#pragma omp for nowait
		for(fint ied = m->gSubDomFaceStart(); ied < m->gSubDomFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);
			const fint jelem = m->gintfac(ied,1);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr(ied,0), ri+ielem*NDIM);
				ufr(ied,i) = linearExtrapolate(u(jelem,i), grads[jelem], i, 1.0,
				                               &gr(ied,0), ri+jelem*NDIM);
			}
		}

#pragma omp for
		for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		{
			const fint ielem = m->gintfac(ied,0);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr(ied,0), ri+ielem*NDIM);
			}
		}
	}
}

template class SolutionReconstruction<freal,NVARS>;
template class SolutionReconstruction<freal,1>;
template class LinearUnlimitedReconstruction<freal,NVARS>;
template class LinearUnlimitedReconstruction<freal,1>;

#ifdef USE_ADOLC
template class SolutionReconstruction<adouble,NVARS>;
template class SolutionReconstruction<adouble,1>;
template class LinearUnlimitedReconstruction<adouble,NVARS>;
template class LinearUnlimitedReconstruction<adouble,1>;
#endif

} // end namespace


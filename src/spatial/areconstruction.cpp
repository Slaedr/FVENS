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
SolutionReconstruction<scalar,nvars>::SolutionReconstruction (const UMesh2dh<scalar> *const mesh,
		const amat::Array2d<scalar>& c_centres,
		const amat::Array2d<scalar>* gauss_r)
	: m{mesh}, ri{c_centres}, gr{gauss_r}, ng{gr[0].rows()}
{ }

template <typename scalar, int nvars>
SolutionReconstruction<scalar,nvars>::~SolutionReconstruction()
{ }

template <typename scalar, int nvars>
LinearUnlimitedReconstruction<scalar,nvars>
::LinearUnlimitedReconstruction(const UMesh2dh<scalar> *const mesh,
                                const amat::Array2d<scalar>& c_centres,
                                const amat::Array2d<scalar>* gauss_r)
	: SolutionReconstruction<scalar,nvars>(mesh, c_centres, gauss_r)
{ }

template <typename scalar, int nvars>
void LinearUnlimitedReconstruction<scalar,nvars>::compute_face_values(
		const MVector<scalar>& u,
		const amat::Array2d<scalar>& ug,
		const GradBlock_t<scalar,NDIM,nvars> *const grads,
		amat::Array2d<scalar>& ufl, amat::Array2d<scalar>& ufr) const
{
#pragma omp parallel default(shared)
	{
#pragma omp for nowait
		for(a_int ied = m->gConnBFaceStart(); ied < m->gConnBFaceEnd(); ied++)
		{
			const a_int ielem = m->gintfac(ied,0);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr[ied](0,0), &ri(ielem,0));
			}
		}

#pragma omp for nowait
		for(a_int ied = m->gSubDomFaceStart(); ied < m->gSubDomFaceEnd(); ied++)
		{
			const a_int ielem = m->gintfac(ied,0);
			const a_int jelem = m->gintfac(ied,1);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr[ied](0,0), &ri(ielem,0));
				ufr(ied,i) = linearExtrapolate(u(jelem,i), grads[jelem], i, 1.0,
				                               &gr[ied](0,0), &ri(jelem,0));
			}
		}

#pragma omp for
		for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
		{
			const a_int ielem = m->gintfac(ied,0);

			for(int i = 0; i < nvars; i++)
			{
				ufl(ied,i) = linearExtrapolate(u(ielem,i), grads[ielem], i, 1.0,
				                               &gr[ied](0,0), &ri(ielem,0));
			}
		}
	}
}

template class SolutionReconstruction<a_real,NVARS>;
template class SolutionReconstruction<a_real,1>;
template class LinearUnlimitedReconstruction<a_real,NVARS>;
template class LinearUnlimitedReconstruction<a_real,1>;

#ifdef USE_ADOLC
template class SolutionReconstruction<adouble,NVARS>;
template class SolutionReconstruction<adouble,1>;
template class LinearUnlimitedReconstruction<adouble,NVARS>;
template class LinearUnlimitedReconstruction<adouble,1>;
#endif

} // end namespace


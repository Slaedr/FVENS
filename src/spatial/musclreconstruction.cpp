/** \file
 * \file Implementation of MUSCL reconstuction schemes
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

#include "musclreconstruction.hpp"

namespace fvens {

template <typename scalar, int nvars>
MUSCLReconstruction<scalar,nvars>::MUSCLReconstruction(const UMesh2dh<scalar> *const mesh,
		const amat::Array2d<scalar>& r_centres, const amat::Array2d<scalar>* gauss_r)
	: SolutionReconstruction<scalar,nvars>(mesh, r_centres, gauss_r), eps{1e-8}, k{1.0/3.0}
{ }

template <typename scalar, int nvars>
inline scalar MUSCLReconstruction<scalar,nvars>::
computeBiasedDifference(const scalar *const ri, const scalar *const rj,
                        const scalar ui, const scalar uj, const scalar *const grads) const
{
	scalar del = 0;
	for(int idim = 0; idim < NDIM; idim++)
		del += grads[idim]*(rj[idim]-ri[idim]);

	return 2.0*del - (uj-ui);
}

template <typename scalar, int nvars> inline
scalar
MUSCLReconstruction<scalar,nvars>::musclReconstructLeft(const scalar ui, const scalar uj,
                                                        const scalar deltam, const scalar phi) const
{
	return ui + phi/4.0*( (1.0-k*phi)*deltam + (1.0+k*phi)*(uj - ui) );
}

template <typename scalar, int nvars> inline
scalar
MUSCLReconstruction<scalar,nvars>::musclReconstructRight(const scalar ui, const scalar uj,
                                                         const scalar deltap, const scalar phi) const
{
	return uj - phi/4.0*( (1.0-k*phi)*deltap + (1.0+k*phi)*(uj - ui) );
}

template <typename scalar, int nvars>
MUSCLVanAlbada<scalar,nvars>::MUSCLVanAlbada(const UMesh2dh<scalar> *const mesh,
		const amat::Array2d<scalar>& r_centres, const amat::Array2d<scalar>* gauss_r)
	: MUSCLReconstruction<scalar,nvars>(mesh, r_centres, gauss_r)
{ }

template <typename scalar, int nvars>
void MUSCLVanAlbada<scalar,nvars>::compute_face_values(const MVector<scalar>& u, 
                                                       const amat::Array2d<scalar>& ug,
                                                       const GradArray<scalar,nvars>& grads,
                                                       amat::Array2d<scalar>& ufl,
                                                       amat::Array2d<scalar>& ufr) const
{
#pragma omp parallel for default(shared)
	for(a_int ied = 0; ied < m->gnbface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < nvars; i++)
		{
			// Note that the copy below is necessary because grads[ielem] is column major
			scalar grad[NDIM];
			for(int j = 0; j < NDIM; j++)
				grad[j] = grads[ielem](j,i);
			
			const scalar deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), ug(ied,i), grad);
			
			scalar phi_l = (2.0*deltam * (ug(ied,i) - u(ielem,i)) + eps) 
				/ (deltam*deltam + (ug(ied,i) - u(ielem,i))*(ug(ied,i) - u(ielem,i)) + eps);
			if( phi_l < 0.0) phi_l = 0.0;

			ufl(ied,i) = musclReconstructLeft(u(ielem,i), ug(ied,i), deltam, phi_l);
		}
	}
	
#pragma omp parallel for default(shared)
	for(a_int ied = m->gnbface(); ied < m->gnaface(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
		const a_int jelem = m->gintfac(ied,1);

		for(int i = 0; i < nvars; i++)
		{
			// Note that the copy below is necessary because grads[ielem] is column major
			scalar gradl[NDIM], gradr[NDIM];
			for(int j = 0; j < NDIM; j++) {
				gradl[j] = grads[ielem](j,i);
				gradr[j] = grads[jelem](j,i);
			}

			const scalar deltam = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradl);
			const scalar deltap = computeBiasedDifference(&ri(ielem,0), &ri(jelem,0),
					u(ielem,i), u(jelem,i), gradr);
			
			scalar phi_l = (2.0*deltam * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltam*deltam + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_l < 0.0) phi_l = 0.0;

			scalar phi_r = (2*deltap * (u(jelem,i) - u(ielem,i)) + eps) 
				/ (deltap*deltap + (u(jelem,i) - u(ielem,i))*(u(jelem,i) - u(ielem,i)) + eps);
			if( phi_r < 0.0) phi_r = 0.0;

			ufl(ied,i) = musclReconstructLeft(u(ielem,i), u(jelem,i), deltam, phi_l);
			ufr(ied,i) = musclReconstructRight(u(ielem,i), u(jelem,i), deltap, phi_r);
		}
	}
}

template class MUSCLVanAlbada<a_real,NVARS>;
template class MUSCLVanAlbada<a_real,1>;

}

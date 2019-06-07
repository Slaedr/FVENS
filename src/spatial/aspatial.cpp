/** @file aspatial.cpp
 * @brief Finite volume spatial discretization
 * @author Aditya Kashi
 * @date Feb 24, 2016
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
#include <iomanip>
#include "aspatial.hpp"
#include "mathutils.hpp"
#include "utilities/aerrorhandling.hpp"
#include "utilities/mpiutils.hpp"
#include "linalg/alinalg.hpp"

namespace fvens {

/** Currently, the ghost cell coordinates are computed as reflections about the face centre.
 * \todo TODO: Replace midpoint-reflected ghost cells with face-reflected ones.
 * \sa compute_ghost_cell_coords_about_midpoint
 * \sa compute_ghost_cell_coords_about_face
 */
template<typename scalar, int nvars>
Spatial<scalar,nvars>::Spatial(const UMesh<scalar,NDIM> *const mesh) : m(mesh)
{
	StatusCode ierr = 0;

	ierr = createGhostedSystemVector(m, NDIM, &rcvec);
	petsc_throw(ierr, "Could not create vec for cell-centres!");

	update_subdomain_cell_centres();
	petsc_throw(ierr, "Could not compute subdomain cell-centres!");

	ierr = VecGhostUpdateBegin(rcvec, INSERT_VALUES, SCATTER_FORWARD);
	petsc_throw(ierr, "Cell centre scatter could not begin!");

	// Compute coords of face centres
	gr.resize(m->gnaface(), NDIM);
	gr.zeros();
	for(fint ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		for(int iv = 0; iv < m->gnnofa(ied); iv++)
			for(int idim = 0; idim < NDIM; idim++)
				gr(ied,idim) += m->gcoords(m->gintfac(ied,2+iv),idim);

		for(int idim = 0; idim < NDIM; idim++)
			gr(ied,idim) /= m->gnnofa(ied);
	}

	ierr = VecGhostUpdateEnd(rcvec, INSERT_VALUES, SCATTER_FORWARD);
	petsc_throw(ierr, "Cell-centre scatter could not be completed!");

	rch.setVec(rcvec);

	rcbp.resize(m->gnbface(),NDIM);
	compute_ghost_cell_coords_about_midpoint(rcbp);
	//compute_ghost_cell_coords_about_face(rchg);

	if(m->gnbface() > 0)
		rcbptr = &rcbp(0,0);
	else
		rcbptr = nullptr;
}

template<typename scalar, int nvars>
Spatial<scalar,nvars>::~Spatial()
{
	rch.restore();
	int ierr = VecDestroy(&rcvec);
	if(ierr) {
		std::cout << "Could not destroy vector of cell centres!\n";
	}
}

template<typename scalar, int nvars>
void Spatial<scalar,nvars>::update_subdomain_cell_centres()
{
	MutableGhostedVecHandler<scalar> rchm(rcvec);
	scalar *const rc = rchm.getArray();

	m->compute_cell_centres(rc);
}

template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_midpoint(amat::Array2d<scalar>& rchg)
{
	const ConstGhostedVecHandler<scalar> rch(rcvec);
	const amat::Array2dView<scalar> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

	for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const fint ielem = m->gintfac(iface,0);

		for(int idim = 0; idim < NDIM; idim++)
		{
			scalar facemidpoint = 0;

			for(int inof = 0; inof < m->gnnofa(iface); inof++)
				facemidpoint += m->gcoords(m->gintfac(iface,2+inof),idim);

			facemidpoint /= m->gnnofa(iface);

			rchg(iface-m->gPhyBFaceStart(),idim) = 2.0*facemidpoint - rc(ielem,idim);
		}
	}
}

/** The ghost cell is a reflection of the boundary cell about the boundary-face.
 * It is NOT the reflection about the midpoint of the boundary-face.
 * \todo TODO Generalize to 3D
 */
template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_face(amat::Array2d<scalar>& rchg)
{
	static_assert(NDIM==2, "Only 2D supported currently!");
	const ConstGhostedVecHandler<scalar> rch(rcvec);
	const amat::Array2dView<scalar> rc(rch.getArray(), m->gnelem()+m->gnConnFace(), NDIM);

	for(fint ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const fint ielem = m->gintfac(ied,0);
		const scalar nx = m->gfacemetric(ied,0);
		const scalar ny = m->gfacemetric(ied,1);

		const scalar xi = rc(ielem,0);
		const scalar yi = rc(ielem,1);

		const scalar x1 = m->gcoords(m->gintfac(ied,2),0);
		const scalar x2 = m->gcoords(m->gintfac(ied,3),0);
		const scalar y1 = m->gcoords(m->gintfac(ied,2),1);
		const scalar y2 = m->gcoords(m->gintfac(ied,3),1);

		// find coordinates of the point on the face that is the midpoint of the line joining
		// the real cell centre and the ghost cell centre
		scalar xs,ys;

		// check if nx != 0 and ny != 0
		if(fabs(nx)>A_SMALL_NUMBER && fabs(ny)>A_SMALL_NUMBER)
		{
			xs = ( yi-y1 - ny/nx*xi + (y2-y1)/(x2-x1)*x1 ) / ((y2-y1)/(x2-x1)-ny/nx);
			//ys = yi + ny/nx*(xs-xi);
			ys = y1 + (y2-y1)/(x2-x1) * (xs-x1);
		}
		else if(fabs(nx)<=A_SMALL_NUMBER)
		{
			xs = xi;
			ys = y1;
		}
		else
		{
			xs = x1;
			ys = yi;
		}
		rchg(ied,0) = 2.0*xs-xi;
		rchg(ied,1) = 2.0*ys-yi;
	}
}

template <typename scalar, int nvars>
void Spatial<scalar,nvars>::
getFaceGradient_modifiedAverage(const scalar *const rcl, const scalar *const rcr,
                                const scalar *const ucl, const scalar *const ucr,
                                const scalar *const gradl, const scalar *const gradr,
                                scalar grad[NDIM][nvars]) const
{
	scalar dr[NDIM], dist=0;
	for(int i = 0; i < NDIM; i++) {
		dr[i] = rcr[i]-rcl[i];
		dist += dr[i]*dr[i];
	}
	dist = sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}

	for(int i = 0; i < nvars; i++)
	{
		scalar davg[NDIM];

		for(int j = 0; j < NDIM; j++)
			davg[j] = 0.5*(gradl[j*nvars+i] + gradr[j*nvars+i]);

		const scalar corr = (ucr[i]-ucl[i])/dist;

		const scalar ddr = dimDotProduct(davg,dr);

		for(int j = 0; j < NDIM; j++)
		{
			grad[j][i] = davg[j] - ddr*dr[j] + corr*dr[j];
		}
	}
}

template <typename scalar, int nvars>
void Spatial<scalar,nvars>
::getFaceGradientAndJacobian_thinLayer(const scalar *const ccleft, const scalar *const ccright,
                                       const freal *const ucl, const freal *const ucr,
                                       const freal *const dul, const freal *const dur,
                                       scalar grad[NDIM][nvars], scalar dgradl[NDIM][nvars][nvars],
                                       scalar dgradr[NDIM][nvars][nvars]) const
{
	scalar dr[NDIM], dist=0;

	for(int i = 0; i < NDIM; i++) {
		dr[i] = ccright[i]-ccleft[i];
		dist += dr[i]*dr[i];
	}
	dist = sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}

	for(int i = 0; i < nvars; i++)
	{
		const scalar corr = (ucr[i]-ucl[i])/dist;        //< The thin layer gradient magnitude

		for(int j = 0; j < NDIM; j++)
		{
			grad[j][i] = corr*dr[j];

			for(int k = 0; k < nvars; k++) {
				dgradl[j][i][k] = -dul[i*nvars+k]/dist * dr[j];
				dgradr[j][i][k] = dur[i*nvars+k]/dist * dr[j];
			}
		}
	}
}

template <typename scalar, int nvars>
StatusCode Spatial<scalar,nvars>::assemble_jacobian(const Vec uvec, Mat A) const
{
	using Eigen::Matrix; using Eigen::RowMajor;

	StatusCode ierr = 0;

	// Get comm to know if this is a serial or parallel assembly
	MPI_Comm mycomm;
	ierr = PetscObjectGetComm((PetscObject)A, &mycomm); CHKERRQ(ierr);
	const int mpisize = get_mpi_size(mycomm);
	const bool isdistributed = (mpisize > 1);

	PetscInt locnelem;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ConstGhostedVecHandler<PetscScalar> uvh(uvec);
	const PetscScalar *const uarr = uvh.getArray();

#pragma omp parallel for default(shared)
	for(fint iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const fint lelem = m->gintfac(iface,0);
		const fint lelemg = isdistributed ? m->gglobalElemIndex(lelem) : lelem;

		Matrix<freal,nvars,nvars,RowMajor> left;
		compute_local_jacobian_boundary(iface, &uarr[lelem*nvars], left);

		// negative L and U contribute to diagonal blocks
		left *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1,&lelemg, 1,&lelemg, left.data(), ADD_VALUES);
		}
	}

#pragma omp parallel for default(shared)
	for(fint iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++)
	{
		const fint lelem = m->gintfac(iface,0);
		const fint relem = m->gintfac(iface,1);
		const fint lelemg = isdistributed ? m->gglobalElemIndex(lelem) : lelem;
		const fint relemg = isdistributed ? m->gglobalElemIndex(relem) : relem;

		Matrix<freal,nvars,nvars,RowMajor> L;
		Matrix<freal,nvars,nvars,RowMajor> U;
		compute_local_jacobian_interior(iface, &uarr[lelem*nvars], &uarr[relem*nvars], L, U);

#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relemg, 1, &lelemg, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelemg, 1, &relemg, U.data(), ADD_VALUES);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0; U *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelemg, 1, &lelemg, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relemg, 1, &relemg, U.data(), ADD_VALUES);
		}
	}

#pragma omp parallel for default(shared)
	for(fint iface = m->gConnBFaceStart(); iface < m->gConnBFaceEnd(); iface++)
	{
		const fint lelem = m->gintfac(iface,0);
		const fint relem = m->gintfac(iface,1);
		const fint lelemg = isdistributed ? m->gglobalElemIndex(lelem) : lelem;
		const fint relemg = isdistributed ? m->gconnface(iface-m->gConnBFaceStart(), 3) : -1;

		Matrix<freal,nvars,nvars,RowMajor> L;
		Matrix<freal,nvars,nvars,RowMajor> U;
		compute_local_jacobian_interior(iface, &uarr[lelem*nvars], &uarr[relem*nvars], L, U);

#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelemg, 1, &relemg, U.data(), ADD_VALUES);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelemg, 1, &lelemg, L.data(), ADD_VALUES);
		}
	}

	return ierr;
}


template class Spatial<freal,NVARS>;
template class Spatial<freal,1>;

}	// end namespace

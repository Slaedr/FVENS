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
#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

/** Currently, the ghost cell coordinates are computed as reflections about the face centre.
 * \todo TODO: Replace midpoint-reflected ghost cells with face-reflected ones.
 * \sa compute_ghost_cell_coords_about_midpoint
 * \sa compute_ghost_cell_coords_about_face
 */
template<typename scalar, int nvars>
Spatial<scalar,nvars>::Spatial(const UMesh2dh<scalar> *const mesh) : m(mesh)
{
	StatusCode ierr = 0;

	// ierr = createGhostedSystemVector(m, NDIM, &rcvec);
	// petsc_throw(ierr, "Could not create vec for cell-centres!");

	rc.resize(m->gnelem()+m->gnConnFace(),NDIM);
	ierr = update_subdomain_cell_centres();
	petsc_throw(ierr, "Could not compute subdomain cell-centres!");

	rcbp.resize(m->gnbface(),NDIM);

	compute_ghost_cell_coords_about_midpoint(rcbp);
	//compute_ghost_cell_coords_about_face(rchg);

	// Compute coords of face centres
	gr.resize(m->gnaface(), NDIM);
	gr.zeros();
	for(a_int ied = m->gFaceStart(); ied < m->gFaceEnd(); ied++)
	{
		for(int iv = 0; iv < m->gnnofa(ied); iv++)
			for(int idim = 0; idim < NDIM; idim++)
				gr(ied,idim) += m->gcoords(m->gintfac(ied,2+iv),idim);

		for(int idim = 0; idim < NDIM; idim++)
			gr(ied,idim) /= m->gnnofa(ied);
	}

	// ierr = VecGhostUpdateEnd(rc, INSERT_VALUES, SCATTER_FORWARD);
	// petsc_throw(ierr, "rc scatter could not be completed!");
}

template<typename scalar, int nvars>
Spatial<scalar,nvars>::~Spatial()
{
	// int ierr = VecDestroy(&rc);
	// if(ierr) {
	// 	std::cout << "Could not destroy vector of cell centres!\n";
	// }
}

template<typename scalar, int nvars>
StatusCode Spatial<scalar,nvars>::update_subdomain_cell_centres()
{
	StatusCode ierr = 0;
	// TODO: Use functions to get ghosted array and abstract AD types and real types
	// Vec localrcvec;
	// ierr = VecGhostGetLocalForm(rcvec, &localrc); CHKERRQ(ierr);
	// scalar *const rcvals = getVecAsArray<scalar>(localrcvec);

	m->compute_cell_centres(&rc(0,0));

	// ierr = VecGhostRestoreLocalForm(rc, &localrc); CHKERRQ(ierr);

	/* Transfer cell centre data from neighboring subdomains for connectivity ghost cells
	 */
	// ierr = VecGhostUpdateBegin(rc, INSERT_VALUES, SCATTER_FORWARD);
	// petsc_throw(ierr, "rc scatter could not begin");

	return ierr;
}

template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_midpoint(amat::Array2d<scalar>& rchg)
{
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);

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
 */
template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_face(amat::Array2d<scalar>& rchg)
{
	static_assert(NDIM==2);
	for(a_int ied = m->gPhyBFaceStart(); ied < m->gPhyBFaceEnd(); ied++)
	{
		const a_int ielem = m->gintfac(ied,0);
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
                                       const a_real *const ucl, const a_real *const ucr,
                                       const a_real *const dul, const a_real *const dur,
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

	PetscInt locnelem; const PetscScalar *uarr;
	ierr = VecGetLocalSize(uvec, &locnelem); CHKERRQ(ierr);
	assert(locnelem % nvars == 0);
	locnelem /= nvars;
	assert(locnelem == m->gnelem());

	ierr = VecGetArrayRead(uvec, &uarr); CHKERRQ(ierr);

#pragma omp parallel for default(shared)
	for(a_int iface = m->gPhyBFaceStart(); iface < m->gPhyBFaceEnd(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);

		Matrix<a_real,nvars,nvars,RowMajor> left;
		compute_local_jacobian_boundary(iface, &uarr[lelem*nvars], left);

		// negative L and U contribute to diagonal blocks
		left *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1,&lelem, 1,&lelem, left.data(), ADD_VALUES);
		}
	}

#pragma omp parallel for default(shared)
	for(a_int iface = m->gSubDomFaceStart(); iface < m->gSubDomFaceEnd(); iface++)
	{
		const a_int lelem = m->gintfac(iface,0);
		const a_int relem = m->gintfac(iface,1);

		Matrix<a_real,nvars,nvars,RowMajor> L;
		Matrix<a_real,nvars,nvars,RowMajor> U;
		compute_local_jacobian_interior(iface, &uarr[lelem*nvars], &uarr[relem*nvars], L, U);

#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &relem, U.data(), ADD_VALUES);
		}

		// negative L and U contribute to diagonal blocks
		L *= -1.0; U *= -1.0;
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &lelem, 1, &lelem, L.data(), ADD_VALUES);
		}
#pragma omp critical
		{
			ierr = MatSetValuesBlocked(A, 1, &relem, 1, &relem, U.data(), ADD_VALUES);
		}
	}

	ierr = VecRestoreArrayRead(uvec, &uarr); CHKERRQ(ierr);

	return ierr;
}


template class Spatial<a_real,NVARS>;
template class Spatial<a_real,1>;

#ifdef USE_ADOLC
template class Spatial<adouble,NVARS>;
template class Spatial<adouble,1>;
#endif
}	// end namespace

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

namespace fvens {

/** Currently, the ghost cell coordinates are computed as reflections about the face centre.
 * \todo TODO: Replace midpoint-reflected ghost cells with face-reflected ones.
 * \sa compute_ghost_cell_coords_about_midpoint
 * \sa compute_ghost_cell_coords_about_face
 */
template<typename scalar, int nvars>
Spatial<scalar,nvars>::Spatial(const UMesh2dh<scalar> *const mesh) : m(mesh)
{
	rc.resize(m->gnelem()+m->gnbface(), NDIM);
	gr = new amat::Array2d<scalar>[m->gnaface()];
	for(int i = 0; i <  m->gnaface(); i++) {
		gr[i].resize(NGAUSS, NDIM);
		gr[i].zeros();
	}

	// get cell centers (real and ghost)
	
	for(a_int ielem = 0; ielem < m->gnelem(); ielem++)
	{
		for(int idim = 0; idim < NDIM; idim++)
		{
			rc(ielem,idim) = 0;
			for(int inode = 0; inode < m->gnnode(ielem); inode++)
				rc(ielem,idim) += m->gcoords(m->ginpoel(ielem, inode), idim);
			rc(ielem,idim) = rc(ielem,idim) / (scalar)(m->gnnode(ielem));
		}
	}

	amat::Array2d<scalar> rchg(m->gnbface(),NDIM);

	compute_ghost_cell_coords_about_midpoint(rchg);
	//compute_ghost_cell_coords_about_face(rchg);

	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		a_int relem = m->gintfac(iface,1);
		for(int idim = 0; idim < NDIM; idim++)
			rc(relem,idim) = rchg(iface,idim);
	}

	//Calculate and store coordinates of Gauss points
	// Gauss points are uniformly distributed along the face.
	/*for(a_int ied = 0; ied < m->gnaface(); ied++)
	{
		scalar x1, y1, x2, y2;
		x1 = m->gcoords(m->gintfac(ied,2),0);
		y1 = m->gcoords(m->gintfac(ied,2),1);
		x2 = m->gcoords(m->gintfac(ied,3),0);
		y2 = m->gcoords(m->gintfac(ied,3),1);
		for(int ig = 0; ig < NGAUSS; ig++)
		{
			gr[ied](ig,0) = x1 + (scalar)(ig+1.0)/(scalar)(NGAUSS+1.0) * (x2-x1);
			gr[ied](ig,1) = y1 + (scalar)(ig+1.0)/(scalar)(NGAUSS+1.0) * (y2-y1);
		}
	}*/

	// Compute coords of face centres (NGAUSS == 1)
	assert(NGAUSS == 1);
	for(a_int ied = 0; ied < m->gnaface(); ied++)
	{
		for(int iv = 0; iv < m->gnnofa(); iv++)
			for(int idim = 0; idim < NDIM; idim++)
				gr[ied](0,idim) += m->gcoords(m->gintfac(ied,2+iv),idim);
			
		for(int idim = 0; idim < NDIM; idim++)
			gr[ied](0,idim) /= m->gnnofa();
	}
}

template<typename scalar, int nvars>
Spatial<scalar,nvars>::~Spatial()
{
	delete [] gr;
}

template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_midpoint(amat::Array2d<scalar>& rchg)
{
	for(a_int iface = 0; iface < m->gnbface(); iface++)
	{
		const a_int ielem = m->gintfac(iface,0);

		for(int idim = 0; idim < NDIM; idim++)
		{
			scalar facemidpoint = 0;
			
			for(int inof = 0; inof < m->gnnofa(); inof++)
				facemidpoint += m->gcoords(m->gintfac(iface,2+inof),idim);
			
			facemidpoint /= m->gnnofa();
			
			rchg(iface,idim) = 2.0*facemidpoint - rc(ielem,idim);
		}
	}
}

/** The ghost cell is a reflection of the boundary cell about the boundary-face.
 * It is NOT the reflection about the midpoint of the boundary-face.
 */
template<typename scalar, int nvars>
void Spatial<scalar,nvars>::compute_ghost_cell_coords_about_face(amat::Array2d<scalar>& rchg)
{
	for(a_int ied = 0; ied < m->gnbface(); ied++)
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
getFaceGradient_modifiedAverage(const a_int iface,
                                const scalar *const ucl, const scalar *const ucr,
                                const scalar *const gradl, const scalar *const gradr,
                                scalar grad[NDIM][nvars]) const
{
	scalar dr[NDIM], dist=0;
	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);

	for(int i = 0; i < NDIM; i++) {
		dr[i] = rc(relem,i)-rc(lelem,i);
		dist += dr[i]*dr[i];
	}
	dist = std::sqrt(dist);
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
void Spatial<scalar,nvars>::getFaceGradientAndJacobian_thinLayer(const a_int iface,
		const a_real *const ucl, const a_real *const ucr,
		const a_real *const dul, const a_real *const dur,
		a_real grad[NDIM][nvars], a_real dgradl[NDIM][nvars][nvars], a_real dgradr[NDIM][nvars][nvars])
	const
{
	a_real dr[NDIM], dist=0;

	const a_int lelem = m->gintfac(iface,0);
	const a_int relem = m->gintfac(iface,1);
	for(int i = 0; i < NDIM; i++) {
		dr[i] = rc(relem,i)-rc(lelem,i);
		dist += dr[i]*dr[i];
	}
	dist = sqrt(dist);
	for(int i = 0; i < NDIM; i++) {
		dr[i] /= dist;
	}

	for(int i = 0; i < nvars; i++) 
	{
		const a_real corr = (ucr[i]-ucl[i])/dist;        //< The thin layer gradient magnitude
		
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

template class Spatial<a_real,NVARS>;
template class Spatial<a_real,1>;

}	// end namespace

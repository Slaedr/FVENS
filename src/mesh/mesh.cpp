/** \file amesh2dh.cpp
 * \brief Implementation data structures and preprocessing of 2D unstructured hybrid grids
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
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "mesh.hpp"
#include "utilities/mpiutils.hpp"

#ifdef USE_ADOLC
#include <adolc/adolc.h>
#endif

namespace fvens {

template <typename scalar, int ndim>
UMesh<scalar,ndim>::UMesh()
	: nconnface{0}, isBoundaryMaps{false}
{  }

template <typename scalar, int ndim>
UMesh<scalar,ndim>::UMesh(const MeshData& md)
	: npoinglobal{md.npoin}, nelemglobal{md.nelem},
	  npoin{md.npoin}, nelem{md.nelem}, nbface{md.nbface}, nnode(md.nnode), maxnnode{md.maxnnode},
	  nfael(md.nfael), maxnfael{md.maxnfael}, nnofa{md.nnofa}, nbtag{md.nbtag}, ndtag{md.ndtag},
	  coords(md.coords), inpoel(md.inpoel), bface(md.bface), vol_regions(md.vol_regions),
	  nconnface{0}
{  }

template <typename scalar, int ndim>
UMesh<scalar,ndim>::~UMesh()
{
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::reorder_cells(const PetscInt *const permvec)
{
	// reorder inpoel, nnode, nfael, vol_regions
	const amat::Array2d<fint> tempelems = inpoel;
	const std::vector<int> tempnnode = nnode;
	const std::vector<int> tempnfael = nfael;

	for(fint i = 0; i < nelem; i++)
	{
		for(int j = 0; j < inpoel.cols(); j++)
			inpoel(i,j) = tempelems(permvec[i],j);
		nnode[i] = tempnnode[permvec[i]];
		nfael[i] = tempnfael[permvec[i]];
	}
}

/**	Stores (in array bpointsb) for each boundary point: the associated global point number and
 * the two bfaces associated with it.
 * Also calculates bfacebp, which is like inpoel for boundary faces -
 * it gives the boundary node number (according to bpointsb) of each local node of a bface.
 * \note Only for linear meshes.
 */
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_boundary_points()
{
	std::cout << "UMesh: compute_boundary_points(): Calculating bpointsb structure"<< std::endl;

	// first, get number of boundary points

	nbpoin = 0;
	amat::Array2d<int > flagb(npoin,1);
	flagb.zeros();
	for(int iface = 0; iface < nbface; iface++)
	{
		for(int inofa = 0; inofa < nnofa; inofa++)
			flagb(bface(iface,inofa)) = 1;
	}
	for(int ipoin = 0; ipoin < npoin; ipoin++)
		nbpoin += flagb(ipoin);

	std::cout << "UMesh: compute_boundary_points(): No. of boundary points = " << nbpoin
		<< std::endl;

	bpointsb.resize(nbpoin,3);
	for(int i = 0; i < nbpoin; i++)
		for(int j = 0; j < 3; j++)
			bpointsb(i,j) = -1;

	bfacebp.resize(nbface,nnofa);

	amat::Array2d<double > lpoin(npoin,1);

	int bp = 0;

	// Next, populate bpointsb by iterating over faces.
	// Also populate bfacebp, which holds the boundary points numbers of the 2 points in a bface.

	lpoin.zeros();		// lpoin will be 1 if the point has been visited
	for(int iface = 0; iface < nbface; iface++)
	{
		int p1, p2;
		p1 = bface(iface,0);
		p2 = bface(iface,1);

		if(lpoin(p1) == 0)	// if this point has not been visited before
		{
			bpointsb(bp,0) = p1;
			bpointsb(bp,2) = iface;
			bfacebp(iface,0) = bp;
			bp++;
			lpoin(p1) = 1;
		}
		else
		{
			// search bpoints for point p1
			int ibp=-1;
			for(int i = 0; i < bp; i++)
			{
				if(bpointsb(i,0) == p1) ibp = i;
			}
			if(ibp==-1) std::cout << "UMesh: compute_boundary_points(): Point not found!"
				<< std::endl;
			bpointsb(ibp,2) = iface;
			bfacebp(iface,0) = ibp;
		}

		if(lpoin(p2) == 0)	// if this point has not been visited before
		{
			bpointsb(bp,0) = p2;
			bpointsb(bp,1) = iface;
			bfacebp(iface,1) = bp;
			bp++;
			lpoin(p2) = 1;
		}
		else
		{
			// search bpoints for point p2
			int ibp=-1;
			for(int i = 0; i < bp; i++)
			{
				if(bpointsb(i,0) == p2) ibp = i;
			}
			if(ibp==-1) std::cout << "UMesh2d: compute_boundary_points(): Point not found!"
				<< std::endl;
			bpointsb(ibp,1) = iface;
			bfacebp(iface,1) = ibp;
		}
	}
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::printmeshstats() const
{
	std::cout << "UMesh: No. of points: " << npoin << ", no. of elements: " << nelem
		<< ", no. of boundary faces " << nbface
		<< ", max no. of nodes per element: " << maxnnode << ", no. of nodes per face: " << nnofa
		<< ", max no. of faces per element: " << maxnfael << std::endl;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::writeGmsh2(const std::string mfile) const
{
	std::cout << "UMesh: writeGmsh2(): writing mesh to file " << mfile << std::endl;
	// decide element type first, based on nfael/nnode and nnofa
	int elm_type = 2;
	int face_type = 1;

	if(nnofa == 3)
		face_type = 8;

	std::ofstream outf;
	open_file_toWrite(mfile, outf);

	outf << std::setprecision(MESHDATA_DOUBLE_PRECISION);
	//std::cout << "nodes\n";
	outf << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
	outf << "$Nodes\n" << npoin << '\n';
	for(int ip = 0; ip < npoin; ip++)
	{
		outf << ip+1;
		for(int j = 0; j < ndim; j++)
			outf << " " << coords(ip,j);
		for(int j = 3-ndim; j > 0; j--)
			outf << " " << 0.0;
		outf << '\n';
	}
	outf << "$EndNodes\n";

	// number of tags should be at least 2
	int nbtagout=nbtag, ndtagout=ndtag;
	if(nbtagout < 2) nbtagout = 2;
	if(ndtagout < 2) ndtagout = 2;
	const int default_tag = 1;

	outf << "$Elements\n" << nelem+nbface << '\n';

	// boundary faces first
	for(int iface = 0; iface < nbface; iface++)
	{
		outf << iface+1 << " " << face_type << " " << nbtagout;
		for(int i = nnofa; i < nnofa+nbtag; i++)    // write tags
			outf << " " << bface(iface,i);
		for(int i = 0; i < nbtagout-nbtag; i++)     // write extra tags if needed
			if(nbtag == 0)
				outf << " " << default_tag;
			else
				// different physical tags must have different elementary tags
				outf << " " << default_tag+bface(iface,nnofa+nbtag-1);

		for(int i = 0; i < nnofa; i++)
			outf << " " << bface(iface,i)+1;		// write nodes
		outf << '\n';
	}
	//std::cout << "elements\n";
	for(int iel = 0; iel < nelem; iel++)
	{
		if(nnode[iel] == 3)
			elm_type = 2;
		else if(nnode[iel] == 4)
			elm_type = 3;
		else if(nnode[iel] == 6)
			elm_type = 9;
		else if(nnode[iel] == 8)
			elm_type = 16;
		else if(nnode[iel]==9)
			elm_type = 10;
		outf << nbface+iel+1 << " " << elm_type << " " << ndtagout;
		for(int i = 0; i < ndtag; i++)
			outf << " " << vol_regions(iel,i);
		for(int i = 0; i < ndtagout - ndtag; i++)      // write at least 2 tags
			if(ndtag == 0)
				outf << " " << default_tag;
			else
				// different physical tags must have different elementary tags
				outf << " " << default_tag + vol_regions(iel,ndtag-1);
		for(int i = 0; i < nnode[iel]; i++)
			outf << " " << inpoel(iel,i)+1;
		outf << '\n';
	}
	outf << "$EndElements\n";

	outf.close();
}

// Computes areas of linear triangles and quads
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_areas()
{
	area.resize(nelem,1);
	for(fint i = 0; i < nelem; i++)
	{
		if(nnode[i] == 3)
			area(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1)
				- gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)
				- gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1)
				- gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
		else if(nnode[i]==4)
		{
			area(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1)
				- gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)
				- gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1)
				- gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
			area(i,0) += 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,2),1)
				- gcoords(ginpoel(i,3),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,2),0)
				- gcoords(ginpoel(i,3),0)) + gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,3),1)
				- gcoords(ginpoel(i,3),0)*gcoords(ginpoel(i,2),1));
		}
	}
}

/// Only computes the subdomain cells' cell centres excludes connectivity ghost cells.
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_cell_centres(scalar *const centres) const
{
	for(fint i = 0; i < nelem; i++)
	{
		for(int idim = 0; idim < ndim; idim++) {
			centres[i*ndim+idim] = 0;
			for(int jnode = 0; jnode < nnode[i]; jnode++)
				centres[i*ndim+idim] += coords(inpoel(i,jnode),idim);
			centres[i*ndim+idim] /= (scalar)nnode[i];
		}
	}
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_topological()
{
#ifdef DEBUG
	std::cout << " UMesh: compute_topological(): Calculating and storing topological info.\n";
#endif

	compute_elementsSurroundingPoints();
	compute_elementsSurroundingElements();
	compute_faceConnectivity();
}

/** Assumption: order of nodes of boundary faces is such that normal points outside,
 * when normal is calculated as
 * 		nx = y2 - y1, ny = -(x2-x1).
 */
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_face_data()
{
	static_assert(ndim==2, "Only 2D is currently supported!");
	//Now compute normals and lengths (only linear meshes!)
	facemetric.resize(naface, 3);
	for(fint i = 0; i < naface; i++)
	{
		facemetric(i,0) = coords(intfac(i,3),1) - coords(intfac(i,2),1);
		facemetric(i,1) = -1.0*(coords(intfac(i,3),0) - coords(intfac(i,2),0));
		facemetric(i,2) = sqrt(pow(facemetric(i,0),2) + pow(facemetric(i,1),2));
		//Normalize the normal vector components
		facemetric(i,0) /= facemetric(i,2);
		facemetric(i,1) /= facemetric(i,2);
	}

#ifdef DEBUG
	std::cout << "UMesh: compute_face_data(): Done.\n";
#endif
}

/// This function is only valid in 2D
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_periodic_map(const int bcm, const int axis)
{
	if(bcm < 0) {
		std::cout << " UMesh: No periodic boundary specified.\n";
		return;
	}
	if(axis < 0) {
		std::cout << " UMesh: No periodic axis specified.\n";
		return;
	}

	// this is used to keep track of faces we've visited
	std::vector<fint> periodicmap(nbface,-1);

	static_assert(ndim == 2, "Periodic map not implemented for 3D!");
	const int ax = 1-axis;  //< The axis along which we'll compare the faces' locations

	/* Whenever we come across a face that's not been processed, we'll set mapped faces
	 * for that face and for the face that it maps to at the same time.
	 */
	for(fint iface = 0; iface < nbface; iface++)
	{
		const fint ifacface = gPhyBFaceStart() + iface;
		if(btags(iface,0) == bcm)
		{
			if(periodicmap[iface] > -1)
				continue;

			// get relevant coordinate of face centre
			const scalar ci = (coords(intfac(ifacface,2),ax)+coords(intfac(ifacface,3),ax))/2.0;

			// Faces before iface have already been paired
			for(fint jface = iface+1; jface < nbface; jface++)
			{
				const fint jfacface = jface + gPhyBFaceStart();
				if(btags(jface,0) == bcm)
				{
					const scalar cj = (coords(intfac(jfacface,2),ax)+coords(intfac(jfacface,3),ax))/2.0;

					// 1e-11 is seemingly the best tolerance Gmsh can offer
					if(fabs(ci-cj) <= 1e-8)
					{
						periodicmap[iface] = jface;
						periodicmap[jface] = iface;

						// the ghost cell at iface is same as the interior cell at jface
						//  and vice versa
						intfac(ifacface,1) = intfac(jfacface,0);
						intfac(jfacface,1) = intfac(ifacface,0);
						break;
					}
				}
			}
		}
	}
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_elementsSurroundingPoints()
{
	esup_p.resize(npoin+1,1);
	esup_p.zeros();

	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nfael[i]; j++)
		{
			/* The first index is inpoel(i,j) + 1 : the + 1 is there because
			 * the storage corresponding to the first node begins at 0, not at 1
			 */
			esup_p(inpoel(i,j)+1,0) += 1;
		}
	}
	// Now make the members of esup_p cumulative
	for(int i = 1; i < npoin+1; i++)
		esup_p(i,0) += esup_p(i-1,0);
	// Now populate esup
	esup.resize(esup_p(npoin,0),1);
	esup.zeros();
	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nfael[i]; j++)
		{
			int ipoin = inpoel(i,j);

			// now put that element no. in the space pointed to by esup_p(ipoin):
			esup(esup_p(ipoin,0),0) = i;

			// an element corresponding to ipoin has been found - increment esup_p for that point:
			esup_p(ipoin,0) += 1;
		}
	}

	//But now esup_p holds increased values:
	// each member increased by the number elements surrounding the corresponding point.
	// So now correct this.
	for(int i = npoin; i >= 1; i--)
		esup_p(i,0) = esup_p(i-1,0);
	esup_p(0,0) = 0;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_elementsSurroundingElements()
{
	esuel.resize(nelem, maxnfael);
	for(int ii = 0; ii < nelem; ii++)
		for(int jj = 0; jj < maxnfael; jj++)
			esuel(ii,jj) = -1;

	amat::Array2d<int > lpoin(npoin,1);
	lpoin.zeros();

	// TODO: Fix for 3D - this would then be different for each face and the kernel below changes
	const int nverfa = 2;      // number of vertices per face

	for(int ielem = 0; ielem < nelem; ielem++)
	{
		amat::Array2d<int > lhelp(nverfa,1);
		lhelp.zeros();

		// first get lpofa for this element
		// lpofa(i,j) holds local vertex number of jth vertex of ith face
		//   (j in [0:nverfa], i in [0:nfael])
		amat::Array2d<int > lpofai(nfael[ielem], nverfa);
		for(int i = 0; i < nfael[ielem]; i++)
		{
			for(int j = 0; j < nverfa; j++)
			{
				// fine as long as operands of % are not negative
				lpofai(i,j) = (i+j) % nnode[ielem];
			}
		}

		for(int ifael = 0; ifael < nfael[ielem]; ifael++)
		{
			for(int i = 0; i < nverfa; i++)
			{
				// lhelp stores global node nos. of vertices of current face of current element
				lhelp(i,0) = inpoel(ielem, lpofai(ifael,i));
				lpoin(lhelp(i,0)) = 1;
			}
			int ipoin = lhelp(0);
			for(int istor = esup_p(ipoin); istor < esup_p(ipoin+1); istor++)
			{
				int jelem = esup(istor);

				if(jelem != ielem)
				{
					// setup lpofa for jelem
					amat::Array2d<int > lpofaj;
					lpofaj.resize(nfael[jelem],nverfa);
					for(int i = 0; i < nfael[jelem]; i++)
						for(int j = 0; j < nverfa; j++)
							lpofaj(i,j) = (i+j)%nfael[jelem];

					for(int jfael = 0; jfael < nfael[jelem]; jfael++)
					{
						//Assume that no. of nodes in face ifael is same as that in face jfael
						int icoun = 0;
						for(int jnofa = 0; jnofa < nverfa; jnofa++)
						{
							int jpoin = inpoel(jelem, lpofaj(jfael,jnofa));
							if(lpoin(jpoin)==1) icoun++;
						}
						if(icoun == nverfa)		// nnofa is 2
						{
							esuel(ielem,ifael) = jelem;
							esuel(jelem,jfael) = ielem;
						}
					}
				}
			}
			for(int i = 0; i < nverfa; i++)
				lpoin(lhelp(i)) = 0;
		}
	}
}

template <typename scalar, int ndim>
EIndex UMesh<scalar,ndim>::getFaceEIndex(const bool phyboundary, const fint iface,
                                       const fint lelem) const
{
	static_assert(ndim==2, "Only 2D is currently supported!");

	EIndex face = -1;

	if(!phyboundary)
		assert(intfac.rows() > 0);

	for(EIndex ifael = 0; ifael < gnfael(lelem); ifael++)
	{
		bool facefound = true;

		// Iterate over each node of the element's face to see if
		//  all nodes of the face iface are matched.
		// Below, gnnofa needs to be replaced by the number of nodes
		//  in the inofa-th face of the element.
		for(FIndex inofa = 0; inofa < gnnofa(iface); inofa++)
		{
			const fint node = ginpoel(lelem,getNodeEIndex(lelem,ifael,inofa));
			bool nodefound = false;
			for(FIndex jnofa = 0; jnofa < gnnofa(iface); jnofa++)
			{
				// NOTE: If iface is a phy boundary face, we use bface. Else intfac must be used.
				if(phyboundary) {
					if(bface(iface,jnofa) == node) {
						nodefound = true;
						break;
					}
				}
				else {
					if(gintfac(iface,jnofa+2) == node) {
						nodefound = true;
						break;
					}
				}
			}

			// If this node of lelem does not any node of iface,
			//  move on to the next face of the lelem.
			if(!nodefound) {
				facefound = false;
				break;
			}
		}

		if(facefound)
		{
			face = ifael;
			break;
		}
	}

	return face;
}

template <typename scalar, int ndim>
std::vector<std::pair<fint,EIndex>> UMesh<scalar,ndim>::compute_phyBFaceNeighboringElements() const
{
	static_assert(ndim==2, "Only 2D is currently supported!");
	std::vector<std::pair<fint,EIndex>> interiorelem(nbface);

	for(fint iface = 0; iface < nbface; iface++)
	{
		// First get sorted list of elements around each point of this face.
		// NOTE: In 3D, nnofa below has to be replaced by nnobfa[iface], the number of nodes per
		//  face for physical boundary faces
		std::vector<std::vector<fint>> nbdelems(nnofa);
		for(int j = 0; j < nnofa; j++)
		{
			const fint point = bface(iface,j);
			for(fint isup_p = esup_p(point); isup_p < esup_p(point+1); isup_p++)
				nbdelems[j].push_back(esup(isup_p));

			std::sort(nbdelems[j].begin(), nbdelems[j].end());  // for using set_intersection below
		}

		// Compute the intersection of the sets of elements surrounding each point of this face.
		//  Initialize with the elements surrounding the first point.
		std::vector<fint> intersection;
		for(unsigned int j = 0; j < nbdelems[0].size(); j++)
			intersection.push_back(nbdelems[0][j]);
		for(int j = 1; j < nnofa; j++)
		{
			std::vector<fint> temp(intersection.size());
			auto it = std::set_intersection(nbdelems[j].begin(), nbdelems[j].end(),
			                                intersection.begin(), intersection.end(), temp.begin());

			assert(it-temp.begin() >= 0);
			assert(it-temp.begin() <= static_cast<ptrdiff_t>(intersection.size()));

			std::copy(temp.begin(), it, intersection.begin());
			intersection.resize(it-temp.begin());
		}

		if(intersection.size() > 1) {
			throw std::logic_error("More than one neighboring element found for bface "
			                       + std::to_string(iface));
		}

		interiorelem[iface].first = intersection[0];

		// find the EIndex of the face
		interiorelem[iface].second = getFaceEIndex(true, iface, intersection[0]);
		assert(interiorelem[iface].second >= 0);
	}

	return interiorelem;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_faceConnectivity()
{
	// calculate number of internal faces
	ninface = 0;
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nfael[ie]; in++)
		{
			int je = esuel(ie,in);
			if(je > ie && je < nelem) {
				ninface++;
			}
		}
	}

	naface = ninface + nbface + nconnface;
	std::cout << "UMesh: compute_faceConnectivity(): Total number of faces= " << naface << std::endl;

	//allocate intfac and elemface
	intfac.resize(naface,nnofa+2);
	elemface.resize(nelem,maxnfael);
	btags.resize(nbface,nbtag);

	// first, phyical boundary faces

	phyBFaceStart = 0;
	phyBFaceEnd = nbface;
	const std::vector<std::pair<fint,EIndex>> intelems = compute_phyBFaceNeighboringElements();

	for(fint iface = 0; iface < nbface; iface++)
	{
		intfac(iface,0) = intelems[iface].first;
		intfac(iface,1) = nelem + nconnface + iface;
		for(FIndex inode = 0; inode < nnofa; inode++)
			intfac(iface,2+inode) = bface(iface,inode);
		for(int j = nnofa; j < nnofa+nbtag; j++)
			btags(iface,j-nnofa) = bface(iface,j);
 
		esuel(intelems[iface].first, intelems[iface].second) = nelem+nconnface+iface;
		elemface(intelems[iface].first, intelems[iface].second) = iface;
	}

	// Next, subdomain interior faces

	subDomFaceStart = nbface;
	subDomFaceEnd = nbface + ninface;

	static_assert(ndim == 2, "Only 2D is currently supported!");   // only remove after generalizing the loop below to 3D

	fint faceindex = nbface;
	//fint faceindex = nbface;

	for(fint ie = 0; ie < nelem; ie++)
	{
		for(EIndex in = 0; in < nnode[ie]; in++)
		{
			const fint je = esuel(ie,in);
			if(je > ie && je < nelem)
			{
				const EIndex in1 = (in+1)%nnode[ie];
				intfac(faceindex,0) = ie;
				intfac(faceindex,1) = je;
				intfac(faceindex,2) = inpoel.get(ie,in);
				intfac(faceindex,3) = inpoel.get(ie,in1);

				elemface(ie,in) = faceindex;
				for(EIndex jnode = 0; jnode < nnode[je]; jnode++)
					if(inpoel.get(ie,in1) == inpoel.get(je,jnode))
						elemface(je,jnode) = faceindex;

				faceindex++;
			}
		}
	}

	assert(faceindex-nbface == ninface);

	// Connectivity faces

	connBFaceStart = nbface+ninface;
	assert(connBFaceStart == faceindex);
	connBFaceEnd = nbface+ninface+nconnface;
	assert(connBFaceEnd == naface);

	for(fint iface = connBFaceStart; iface < connBFaceEnd; iface++)
	{
		const fint icface = iface - connBFaceStart;
		const fint inelem = connface(icface,0);
		intfac(iface,0) = inelem;
		esuel(inelem,connface(icface,1)) = nelem+icface;
		elemface(inelem,connface(icface,1)) = iface;
		intfac(iface,1) = nelem+icface;

		// Note that the face always 'points outside' the subdomain, because it always 
		//  'points outside' the cell.
		for(FIndex inode = 0; inode < nnofa; inode++)
			intfac(iface,2+inode) = inpoel(inelem,getNodeEIndex(inelem, connface(icface,1), inode));
	}

	domFaceStart = nbface;
	domFaceEnd = nbface + nconnface + ninface;
	assert(domFaceEnd == naface);
}

template <typename scalar, int ndim>
std::vector<fint> UMesh<scalar,ndim>::getConnectivityGlobalIndices() const
{
	std::vector<fint> cgind(nconnface);
	for(fint iface = 0; iface < nconnface; iface++)
		cgind[iface] = connface(iface,3);

	return cgind;
}

/** \todo: There is an issue with psup for some boundary nodes
 * belonging to elements of different types. Correct this.
 */
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_pointsSurroundingPoints()
{
#ifdef DEBUG
	std::cout << "UMesh: compute_topological(): Points surrounding points\n";
#endif
	psup_p.resize(npoin+1,1);
	psup_p.zeros();
	psup_p(0,0) = 0;

	// The ith member indicates the global point number of which
	// the ith point is a surrounding point
	amat::Array2d<int > lpoin(npoin,1);
	for(int i = 0; i < npoin; i++) lpoin(i,0) = -1;	// initialize this std::vector to -1
	int istor = 0;

	// first pass: calculate storage needed for psup
	for(fint ip = 0; ip < npoin; ip++)
	{
		lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
		// Loop over elements surrounding this point
		for(fint ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
		{
			fint ielem = esup(ie,0);		// element number

			// find local node number of ip in ielem
			int inode = -1;
			for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				if(inpoel(ielem,jnode) == ip) inode = jnode;
#ifdef DEBUG
			if(inode == -1) {
				std::cout << " ! UMesh: compute_topological(): ";
				std::cout << "inode not found while computing psup!\n";
			}
#endif

			// contains true if that local node number is connected to a particular local node.
			std::vector<bool> nbd(nnode[ielem]);
			for(int j = 0; j < nnode[ielem]; j++)
				nbd[j] = false;

			if(nnode[ielem] == 3)
				for(size_t i = 0; i < nbd.size(); i++)
					nbd[i] = true;
			else if(nnode[ielem] == 4)
				for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				{
					if(jnode == (inode + 1) % nnode[ielem]
							|| jnode == (inode + nnode[ielem]-1) % nnode[ielem])
						nbd[jnode] = true;
				}

			//loop over nodes of the element
			for(int inode = 0; inode < nnode[ielem]; inode++)
			{
				//Get global index of this node
				int jpoin = inpoel(ielem, inode);

				/* test if this point as already been counted as a surrounding point of ip,
				 * and whether it's connected to ip.
				 */
				if(lpoin(jpoin,0) != ip && nbd[inode])
				{
					istor++;
					lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
				}
			}
		}
		psup_p(ip+1,0) = istor;
	}

	psup.resize(istor,1);

	//second pass: populate psup
	istor = 0;
	for(fint i = 0; i < npoin; i++)
		lpoin(i,0) = -1;
	for(fint ip = 0; ip < npoin; ip++)
	{
		lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
		// Loop over elements surrounding this point
		for(int ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
		{
			fint ielem = esup(ie,0);		// element number

			// find local node number of ip in ielem
			int inode = -1;
			for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				if(inpoel(ielem,jnode) == ip) inode = jnode;
#ifdef DEBUG
			if(inode == -1) {
				std::cout << " ! UMesh: compute_topological(): ";
				std::cout << "inode not found while computing psup!\n";
			}
#endif

			// nbd[j] contains true if ip is connected to local node number j of ielem.
			std::vector<bool> nbd(nnode[ielem]);
			for(int j = 0; j < nnode[ielem]; j++)
				nbd[j] = false;

			if(nnode[ielem] == 3)
				for(size_t i = 0; i < nbd.size(); i++)
					nbd[i] = true;
			else if(nnode[ielem] == 4)
				for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				{
					if(jnode == (inode + 1) % nnode[ielem]
							|| jnode == (inode + nnode[ielem]-1) % nnode[ielem])
						nbd[jnode] = true;
				}

			//loop over nodes of the element
			for(int inode = 0; inode < nnode[ielem]; inode++)
			{
				//Get global index of this node
				int jpoin = inpoel(ielem, inode);
				if(lpoin(jpoin,0) != ip && nbd[inode])
					// test of this point as already been counted as a surrounding point of ip
				{
					psup(istor,0) = jpoin;
					istor++;
					lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
				}
			}
		}
	}
}

template class UMesh<freal,2>;

} // end namespace

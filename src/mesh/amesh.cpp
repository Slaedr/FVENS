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
#include <boost/algorithm/string.hpp>
#include "amesh2dh.hpp"
#include "utilities/aoptionparser.hpp"

namespace fvens {

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::triNnofa = 2;
template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::quadNnofa = 2;
template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::tetNnofa = 3;
template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::pyrNnofa[5] = {4,3,3,3,3};
template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::prismNnofa[5] = {3,3,4,4,4};
template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::hexNnofa = 4;

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::triLFM[3][2] = {{0,1}, {1,2}, {2,0}};

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::quadLFM[4][2] = {
	{0,1}, {1,2}, {2,3}, {3,0}
};

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::tetLFM[4][3] = {
	{1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}
};

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::pyrLFM[5][4] = {
	{0,3,2,1}, {0,1,4,-1}, {1,2,4,-1}, {3,4,2,-1}, {0,4,3,-1}
};

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::prismLFM[5][4] = {
	{0,2,1}, {5,3,4}, {1,2,5,4}, {0,3,5,2}, {0,1,4,3}
};

template <typename scalar, int ndim>
const int UMesh<scalar,ndim>::hexLFM[6][4] = {
	{0,3,2,1}, {4,5,6,7}, {0,1,5,4}, {3,7,6,2}, {1,2,6,5}, {0,4,7,3}
};

template <typename scalar, int ndim>
UMesh<scalar,ndim>::UMesh() 
	: isBoundaryMaps{false}
{  }

template <typename scalar, int ndim>
UMesh<scalar,ndim>::~UMesh()
{
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::readMesh(const std::string mfile)
{
	std::vector<std::string> parts;
	boost::split(parts, mfile, boost::is_any_of("."));

	if(parts[parts.size()-1] == "su2")
		readSU2(mfile);
	else
		readGmsh2(mfile);
}

/// Reads mesh from Gmsh 2 format file
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::readGmsh2(const std::string mfile)
{
	int dum; double dummy; std::string dums; char ch;

	std::ifstream infile;
	open_file_toRead(mfile, infile);
	
	for(int i = 0; i < 4; i++)		//skip 4 lines
		do
			ch = infile.get();
		while(ch != '\n');

	infile >> npoin;
	coords.resize(npoin,ndim);

	// read coords of points
	for(int i = 0; i < npoin; i++)
	{
		infile >> dum;
		for(int j = 0; j < ndim; j++)
			infile >> coords(i,j);
		if(ndim < 3) infile >> dummy;
	}
	infile >> dums;		// get 'endnodes'
	infile >> dums;		// get 'elements'

	int width_elms = 25;
	int nelm, elmtype, nbtags, ntags;
	/// elmtype is the standard element type in the Gmsh 2 mesh format - of either faces or elements
	ndtag = 0; nbtag = 0;
	infile >> nelm;
	amat::Array2d<a_int > elms(nelm,width_elms);
	nface = 0; nelem = 0;
	std::vector<int> nnodes(nelm,0);
	std::vector<int> nfaels(nelm,0);
	//std::cout << "UMesh2d: readGmsh2(): Total number of elms is " << nelm << std::endl;

	for(int i = 0; i < nelm; i++)
	{
		infile >> dum;
		infile >> elmtype;
		/** elmtype is different for all faces and for all elements. 
		 * However, meshes in which high-order and linear elements are both present 
		 * are not supported.
		 */
		switch(elmtype)
		{
			case(1): // linear edge
				nnofa = 2;
				infile >> nbtags;
				if(nbtags > nbtag) nbtag = nbtags;
				for(int j = 0; j < nbtags; j++)
					infile >> elms(i,j+nnofa);		// get tags
				for(int j = 0; j < nnofa; j++)
					infile >> elms(i,j);			// get node numbers
				nface++;
				break;
			case(8): // quadratic edge
				nnofa = 3;
				infile >> nbtags;
				if(nbtags > nbtag) nbtag = nbtags;
				for(int j = 0; j < nbtags; j++)
					infile >> elms(i,j+nnofa);		// get tags
				for(int j = 0; j < nnofa; j++)
					infile >> elms(i,j);			// get node numbers
				nface++;
				break;
			case(2): // linear triangles
				nnodes[i] = 3;
				nfaels[i] = 3;
				nnofa = 2;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
				break;
			case(3):	// linear quads
				nnodes[i] = 4;
				nfaels[i] = 4;
				nnofa = 2;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
				break;
			case(9):	// quadratic triangles
				nnodes[i] = 6;
				nfaels[i] = 3;
				nnofa = 3;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
				break;
			case(16):	// quadratic quad (8 nodes)
				nnodes[i] = 8;
				nfaels[i] = 4;
				nnofa = 3;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
				break;
			case(10):	// quadratic quad (9 nodes)
				nnodes[i] = 9;
				nfaels[i] = 4;
				nnofa = 3;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
				break;
			default:
				std::cout << "! UMesh2d: readGmsh2(): Element type not recognized.";
				std::cout << " Setting as linear triangle." << std::endl;
				nnodes[i] = 3;
				nfaels[i] = 3;
				nnofa = 2;
				infile >> ntags;
				if(ntags > ndtag) ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				nelem++;
		}
	}
	/*std::cout << "UMesh2d: readGmsh2(): Done reading elms" << std::endl;
	for(int i = 0; i < nelm; i++)
		std::cout << nnodes[i] << " " << nfaels[i] << std::endl;*/

	nnode.reserve(nelem);
	nfael.reserve(nelem);

	// calculate max nnode and nfael
	maxnnode = nnodes[nface]; maxnfael = nfaels[nface];
	for(int i = 0; i < nelm; i++)
	{
		if(nnodes[i] > maxnnode)
			maxnnode = nnodes[i];
		if(nfaels[i] > maxnfael)
			maxnfael = nfaels[i];
	}

	if(nface > 0)
		bface.resize(nface, nnofa+nbtag);
	else std::cout << "UMesh2d: readGmsh2(): NOTE: There is no boundary data!" << std::endl;

	inpoel.resize(nelem, maxnnode);
	vol_regions.resize(nelem, ndtag);

	std::cout << "UMesh: readGmsh2(): No. of points: " << npoin 
		<< ", number of elements: " << nelem 
		<< ",\nnumber of boundary faces " << nface 
		<< ", max no. of nodes per element: " << maxnnode 
		<< ",\nno. of nodes per face: " << nnofa 
		<< ", max faces per element: " << maxnfael << std::endl;

	// write into inpoel and bface
	// the first nface rows to be read are boundary faces
	for(int i = 0; i < nface; i++)
	{
		for(int j = 0; j < nnofa; j++)
			// -1 to correct for the fact that our numbering starts from zero
			bface(i,j) = elms(i,j)-1;			
		for(int j = nnofa; j < nnofa+nbtag; j++)
			bface(i,j) = elms(i,j);
	}
	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nnodes[i+nface]; j++)
			inpoel(i,j) = elms(i+nface,j)-1;
		for(int j = 0; j < ndtag; j++)
			vol_regions(i,j) = elms(i+nface,j+nnodes[i+nface]);
		nnode.push_back(nnodes[i+nface]);
		nfael.push_back(nfaels[i+nface]);
	}
	infile.close();
	
	// set flag_bpoin
	flag_bpoin.resize(npoin,1);
	flag_bpoin.zeros();
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < nnofa; j++)
			flag_bpoin(bface(i,j)) = 1;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::readSU2(const std::string mfile)
{
	std::string dum;
	std::ifstream fin;
	open_file_toRead(mfile, fin);

	std::getline(fin, dum, '='); std::getline(fin,dum);
	int nfiledim = std::stoi(dum);
	if(nfiledim != ndim)
		std::cout << "! UMesh: readSU2: Mesh is not " << ndim << "-dimensional!\n";

	// read element node connectivity

	std::getline(fin, dum, '='); std::getline(fin,dum);
	nelem = std::stoi(dum);
	std::cout << "UMesh: readSU2: Number of elements = " << nelem << std::endl;

	// Let's just assume a hybrid grid with triangles and quads
	maxnnode = 4; maxnfael = 4;
	inpoel.resize(nelem,maxnnode);
	nnode.resize(nelem); nfael.resize(nelem);

	for(a_int iel = 0; iel < nelem; iel++)
	{
		int id, ddum;
		fin >> id;
		switch(id) 
		{
			case 5: // triangle
				nnode[iel] = 3;
				nfael[iel] = 3;
				break;
			case 9: // quad
				nnode[iel] = 4;
				nfael[iel] = 4;
				break;
			default:
				std::cout << "! UMesh: readSU2: Unknown element type!!\n";
		}

		for(int i = 0; i < nnode[iel]; i++)
			fin >> inpoel(iel,i);
		fin >> ddum;
	}
	// clear the newline
	std::getline(fin,dum);

	// read coordinates of nodes

	std::getline(fin, dum, '='); std::getline(fin,dum);
	npoin = std::stoi(dum);
#ifdef DEBUG
	std::cout << "UMesh: readSU2: Number of points = " << npoin << std::endl;
#endif

	coords.resize(npoin,ndim);
	for(a_int ip = 0; ip < npoin; ip++)
	{
		int ddum;
		for(int j = 0; j < ndim; j++)
			fin >> coords(ip,j);
		fin >> ddum;
	}
	// clear the newline
	std::getline(fin,dum);

	// read boundary face data

	nbtag = 1; ndtag = 0;

	std::getline(fin, dum, '='); std::getline(fin,dum);
	int nbmarkers = std::stoi(dum);
#ifdef DEBUG
	std::cout << "UMesh: readSU2: Number of BC markers = " << nbmarkers << std::endl;
#endif
	
	std::vector<std::vector<std::vector<a_int>>> bfacs(nbmarkers);
	
	std::vector<int> tags(nbmarkers);
	std::vector<a_int> numfacs(nbmarkers);
	nface = 0;

	for(int ib = 0; ib < nbmarkers; ib++)
	{
		std::getline(fin, dum, '='); std::getline(fin,dum);
		tags[ib] = std::stoi(dum);
		std::getline(fin, dum, '='); std::getline(fin,dum);
		numfacs[ib] = std::stoi(dum);
		nface += numfacs[ib];

		bfacs[ib].resize(numfacs[ib]);
		nnofa = 2;

		for(a_int iface = 0; iface < numfacs[ib]; iface++)
		{
			bfacs[ib][iface].resize(nnofa);
			int ddum;
			fin >> ddum;
			for(int inofa = 0; inofa < nnofa; inofa++)
				fin >> bfacs[ib][iface][inofa];
		}
		
		//clear newline, except the last one because it may not exist
		if(ib < nbmarkers-1)
			std::getline(fin,dum);
	}

	fin.close();

	std::cout << "UMesh: readSU2: Number of boundary faces = " << nface << std::endl;

	bface.resize(nface,nnofa+nbtag);
	a_int count=0;
	for(int ib = 0; ib < nbmarkers; ib++)
	{
		for(a_int iface = 0; iface < numfacs[ib]; iface++) 
		{
			for(int inofa = 0; inofa < nnofa; inofa++)
				bface(count,inofa) = bfacs[ib][iface][inofa];
			bface(count,nnofa) = tags[ib];
			
			count++;
		}
	}
	
	// Classify points as boundary points or not
	
	flag_bpoin.resize(npoin,1);
	flag_bpoin.zeros();
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < nnofa; j++)
			flag_bpoin(bface(i,j)) = 1;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::reorder_cells(const PetscInt *const permvec)
{
	// reorder inpoel, nnode, nfael, vol_regions
	const amat::Array2d<a_int> tempelems = inpoel;
	const std::vector<int> tempnnode = nnode;
	const std::vector<int> tempnfael = nfael;
	
	for(a_int i = 0; i < nelem; i++)
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
	for(int iface = 0; iface < nface; iface++)
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

	bfacebp.resize(nface,nnofa);
	
	amat::Array2d<double > lpoin(npoin,1);

	int bp = 0;

	// Next, populate bpointsb by iterating over faces. 
	// Also populate bfacebp, which holds the boundary points numbers of the 2 points in a bface.
	
	lpoin.zeros();		// lpoin will be 1 if the point has been visited
	for(int iface = 0; iface < nface; iface++)
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
		<< ", no. of boundary faces " << nface 
		<< ", max no. of nodes per element: " << maxnnode << ", no. of nodes per face: " << nnofa 
		<< ", max no. of faces per element: " << maxnfael << std::endl;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::writeGmsh2(const std::string mfile)
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

	outf << "$Elements\n" << nelem+nface << '\n';

	// boundary faces first
	for(int iface = 0; iface < nface; iface++)
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
		outf << nface+iel+1 << " " << elm_type << " " << ndtagout;
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

// Computes volumes of linear triangles and quads
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_volumes()
{
	volume.resize(nelem,1);
	for(a_int i = 0; i < nelem; i++)
	{
		if(nnode[i] == 3)
			volume(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) 
				- gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0) 
				- gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) 
				- gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
		else if(nnode[i]==4)
		{
			volume(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) 
				- gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)
				- gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) 
				- gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
			volume(i,0) += 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,2),1) 
				- gcoords(ginpoel(i,3),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,2),0)
				- gcoords(ginpoel(i,3),0)) + gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,3),1) 
				- gcoords(ginpoel(i,3),0)*gcoords(ginpoel(i,2),1));
		}
	}
}
	
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_cell_centres(std::vector<scalar>& centres) const
{
	for(a_int i = 0; i < nelem; i++)
	{
		for(int idim = 0; idim < ndim; idim++) {
			centres[i*ndim+idim] = 0;
			for(int j = 0; j < nnode[i]; j++)
				centres[i*ndim+idim] += coords(inpoel(i,j),idim);
			centres[i*ndim+idim] /= nnode[i];
		}
	}
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_topological()
{
#ifdef DEBUG
	std::cout << "UMesh: compute_topological(): Calculating and storing topological info...\n";
#endif

	compute_elementsSurroundingPoints();
	compute_elementsSurroundingElements();
	compute_faceConnectivity();

#ifdef DEBUG
	std::cout << "UMesh: compute_topological(): Done." << std::endl;
#endif
}

/** Assumption: order of nodes of boundary faces is such that normal points outside, 
 * when normal is calculated as
 * 		nx = y2 - y1, ny = -(x2-x1).
 */
template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_face_data()
{
	int i, j, p1, p2;

	//Now compute normals and lengths (only linear meshes!)
	facemetric.resize(naface, 3);
	for(i = 0; i < naface; i++)
	{
		facemetric(i,0) = coords(intfac(i,3),1) - coords(intfac(i,2),1);
		facemetric(i,1) = -1.0*(coords(intfac(i,3),0) - coords(intfac(i,2),0));
		facemetric(i,2) = sqrt(pow(facemetric(i,0),2) + pow(facemetric(i,1),2));
		//Normalize the normal vector components
		facemetric(i,0) /= facemetric(i,2);
		facemetric(i,1) /= facemetric(i,2);
	}

	//Populate boundary flags in intfacbtags
#ifdef DEBUG
	std::cout << "UTriMesh: compute_face_data(): Storing boundary flags in intfacbtags...\n";
#endif
	intfacbtags.resize(nbface,nbtag);
	for(int ied = 0; ied < nbface; ied++)
	{
		p1 = intfac(ied,2);
		p2 = intfac(ied,3);
		
		if(nbface != nface) { 
			std::cout <<"UMesh: Calculation of number of boundary faces is wrong!" << std::endl; 
			break; 
		}
		for(i = 0; i < nface; i++)
		{
			if(bface(i,0) == p1 || bface(i,1) == p1)
			{
				if(bface(i,1) == p2 || bface(i,0) == p2)
				{
					for(j = 0; j < nbtag; j++)
					{
						intfacbtags(ied,j) = bface.get(i,nnofa+j);
						intfacbtags(ied,j) = bface.get(i,nnofa+j);
					}
				}
			}
		}
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
	std::vector<a_int> periodicmap(nbface,-1);
	
	const int ax = 1-axis;  //< The axis along which we'll compare the faces' locations

	/* Whenever we come across a face that's not been processed, we'll set mapped faces
	 * for that face and for the face that it maps to at the same time.
	 */
	for(a_int iface = 0; iface < nbface; iface++)
	{
		if(intfacbtags(iface,0) == bcm) 
		{
			if(periodicmap[iface] > -1)
				continue;

			// get relevant coordinate of face centre
			const a_real ci = (coords(intfac(iface,2),ax)+coords(intfac(iface,3),ax))/2.0;

			// Faces before iface have already been paired
			for(a_int jface = iface+1; jface < nbface; jface++)
			{
				if(intfacbtags(jface,0) == bcm) 
				{
					const a_real cj = (coords(intfac(jface,2),ax)+coords(intfac(jface,3),ax))/2.0;
					
					// 1e-11 is seemingly the best tolerance Gmsh can offer
					if(std::fabs(ci-cj) <= 1e-8)
					{
						periodicmap[iface] = jface;
						periodicmap[jface] = iface;

						// the ghost cell at iface is same as the interior cell at jface
						//  and vice versa
						intfac(iface,1) = intfac(jface,0);
						intfac(jface,1) = intfac(iface,0);
						break;
					}
				}
			}
		}
	}
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_boundary_maps()
{
	// iterate over bfaces and find corresponding intfac face for each bface
	bifmap.resize(nbface,1);
	ifbmap.resize(nbface,1);

	std::vector<int> fpo(nnofa);

	for(int ibface = 0; ibface < nface; ibface++)
	{
		for(int i = 0; i < nnofa; i++)
			fpo[i] = bface(ibface,i);

		int inface = -1;

		// iterate over intfacs - check if bface ibface matches intfac iface, for each iface
		for(int iface = 0; iface < nbface; iface++)
		{
			bool final1 = true;

			std::vector<bool> inter(nnofa);
			for(int b = 0; b < nnofa; b++)
				inter[b] = false;						// initially set all bools to false

			for(int j = 0; j < nnofa; j++)
			{
				for(int k = 0; k < nnofa; k++)
					if(fpo[j] == intfac(iface, 2+k)) {
						/* if jth node of ibface has a node of iface, it belongs to iface; 
						 * set the corresp. boolean to true
						 */
						inter[j] = true;
						break;
					}
			}

			/*for(int i = 0; i < nnofa; i++)
				std::cout << inter[i];
			std::cout << std::endl;*/

			for(int b = 0; b < nnofa; b++)
				/* if any node of ibface failed to find a node of iface, 
				 * ibface is not the same as iface
				 */
				if(inter[b] == false) final1 = false;	

			if(final1 == true) inface = iface;
		}

		if(inface != -1) {
			bifmap(inface) = ibface;
			ifbmap(ibface) = inface;
		}
		else {
			std::cout << "UMesh2d: compute_boundary_maps(): ! intfac face corresponding to " 
				<< ibface << "th bface not found!!" << std::endl;
			continue;
		}
	}
	isBoundaryMaps = true;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::writeBoundaryMapsToFile(std::string mapfile)
{
	if(isBoundaryMaps == false) {
		std::cout << "UMesh2d: writeBoundaryMapsToFile(): ! Boundary maps not available!" 
			<< std::endl;
		return;
	}
	std::ofstream ofile(mapfile);
	ofile << nbface << '\n'<< "bifmap\n";
	for(int i = 0; i < nbface; i++)
		ofile << bifmap.get(i) << ' ';
	ofile << '\n';
	ofile << "ifbmap\n";
	for(int i = 0; i < nbface; i++)
		ofile << ifbmap.get(i) << ' ';
	ofile << '\n';
	ofile.close();
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::readBoundaryMapsFromFile(std::string mapfile)
{
	std::ifstream ofile(mapfile);
	std::string dum; int sz;
	ofile >> sz >>  dum;
	std::cout << "UMesh2d: readBoundaryMapsFromFile(): Number of boundary faces in file = " 
		<< sz << std::endl;
	bifmap.resize(sz,1);
	ifbmap.resize(sz,1);

	for(int i = 0; i < nbface; i++)
		ofile >> bifmap(i);

	ofile >> dum;
	for(int i = 0; i < nbface; i++)
		ofile >> ifbmap(i);

	ofile.close();
	isBoundaryMaps = true;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_intfacbtags()
{
	/// Populate intfacbtags with boundary markers of corresponding bfaces

	intfacbtags.resize(nface,nbtag);

	if(isBoundaryMaps == false)
	{
		std::cout << "UMesh2d: compute_intfacbtags(): ! Boundary maps are not available!" 
			<< std::endl;
		return;
	}

	for(int ibface = 0; ibface < nface; ibface++)
	{
		for(int j = 0; j < nbtag; j++)
			intfacbtags(ifbmap(ibface),j) = bface(ibface,nnofa+j);
	}
}

/**	Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
 * NOTE: Make sure to execute [compute_topological()](@ref compute_topological) 
 * before calling this function.
 */
template <typename scalar, int ndim>
UMesh<scalar,ndim> UMesh<scalar,ndim>::convertLinearToQuadratic()
{
	std::cout << "UMesh2d: convertLinearToQuadratic(): Producing quadratic mesh from linear mesh" 
		<< std::endl;
	UMesh<scalar,ndim> q;
	if(nnofa != 2) 
	{ 
		std::cout << "! UMesh2d: convertLinearToQuadratic(): Mesh is not linear!!" 
		<< std::endl; return q;
	}

	int parm = 1;		// 1 extra node per face
	
	/** We first calculate: 
	 * total number of non-simplicial elements; 
	 * nnode, nfael in each element; 
	 * mmax nnode and max nfael.
	 */
	int nelemnonsimp = 0;		// total number of non-simplicial elements
	
	q.nnode.resize(nelem);		// allocate
	q.nfael.resize(nelem);

	q.maxnfael = maxnfael;
	q.maxnnode = 0; 
	for(int ielem = 0; ielem < nelem; ielem++)
	{
		q.nfael[ielem] = nfael[ielem];
		
		if(nnode[ielem] >= 4) 	// if mesh is not simplicial
		{
			nelemnonsimp++;
			q.nnode[ielem] = nnode[ielem] + nfael[ielem]*parm + 1;
			if(q.nnode[ielem] > q.maxnnode)
				q.maxnnode = q.nnode[ielem];
		}
		else
		{
			q.nnode[ielem] = nnode[ielem] + nfael[ielem]*parm;
			if(q.nnode[ielem] > q.maxnnode)
				q.maxnnode = q.nnode[ielem];
		}	
	}

	q.npoin = npoin + naface + nelemnonsimp;
	q.nelem = nelem;
	q.nface = nface;
	q.nbface = nbface;
	q.naface = naface;
	q.nnofa = nnofa+parm;
	q.nbtag = nbtag;
	q.ndtag = ndtag;

	q.coords.resize(q.npoin, ndim);
	q.inpoel.resize(q.nelem, q.maxnnode);
	q.bface.resize(q.nface, q.nnofa+q.nbtag);

	/// Next, we copy over low-order mesh data to the new mesh.
	for(int i = 0; i < npoin; i++)
		for(int j = 0; j < ndim; j++)
			q.coords(i,j) = coords(i,j);

	for(int i = 0; i < nelem; i++)
		for(int j = 0; j < nnode[i]; j++)
			q.inpoel(i,j) = inpoel(i,j);

	for(int i = 0; i < nface; i++)
	{
		for(int j = 0; j < nnofa; j++)
			q.bface(i,j) = bface(i,j);
		for(int j = nnofa; j < nnofa+nbtag; j++)
			q.bface(i,j+parm) = bface(i,j);
	}

	q.vol_regions = vol_regions;

	/// We then iterate over faces, introducing the required number of points in each face.
	
	// iterate over boundary faces
	for(int ied = 0; ied < nbface; ied++)
	{
		int ielem = intfac(ied,0);
		int p1 = intfac(ied,2);
		int p2 = intfac(ied,3);
		int lp1 = -100000;

		for(int idim = 0; idim < ndim; idim++)
			q.coords(npoin+ied*parm,idim) = (coords(p1,idim) + coords(p2,idim))/2.0;

		for(int inode = 0; inode < nnode[ielem]; inode++)
		{
			if(p1 == inpoel(ielem,inode)) lp1 = inode;
			//if(p2 == inpoel(ielem,inode)) lp2 = inode;
		}

		// in the left element, the new point is in face ip1 
		// (ie, the face whose first point is ip1 in CCW order)
		q.inpoel(ielem, nnode[ielem]+lp1) = npoin+ied*parm;

		// find the bface that this face corresponds to
		for(int ifa = 0; ifa < nface; ifa++)
		{
			if((p1 == bface(ifa,0) && p2 == bface(ifa,1)) 
					|| (p1 == bface(ifa,1) && p2 == bface(ifa,0)))	// face found
			{
				q.bface(ifa,nnofa) = npoin+ied*parm;
			}
		}
	}

	// iterate over internal faces
	for(int ied = nbface; ied < naface; ied++)
	{
		int ielem = intfac(ied,0);
		int jelem = intfac(ied,1);
		int p1 = intfac(ied,2);
		int p2 = intfac(ied,3);
		int lp1 = -100000;
		int lp2 = -100000;

		for(int idim = 0; idim < ndim; idim++)
			q.coords(npoin+ied*parm,idim) = (coords(p1,idim) + coords(p2,idim))/2.0;

		// First look at left element
		for(int inode = 0; inode < nnode[ielem]; inode++)
		{
			if(p1 == inpoel(ielem,inode)) lp1 = inode;
			if(p2 == inpoel(ielem,inode)) lp2 = inode;
		}

		q.inpoel(ielem, nnode[ielem]+lp1) = npoin+ied*parm;

		// Then look at right element
		for(int inode = 0; inode < nnode[jelem]; inode++)
		{
			if(p1 == inpoel(jelem,inode)) lp1 = inode;
			if(p2 == inpoel(jelem,inode)) lp2 = inode;
		}

		// in the right element, the new point is in face ip2
		q.inpoel(jelem, nnode[jelem]+lp2) = npoin+ied*parm;
	}
	
	// for non-simplicial mesh, add extra points at cell-centres as well

	int numpoin = npoin+naface*parm;		// next global point number to be added
	// get cell centres
	for(int iel = 0; iel < nelem; iel++)
	{
		//parmcell = 1;		// number of extra nodes per cell in the interior of the cell
		double c_x = 0, c_y = 0;

		if(nnode[iel] == 4)	
		{
			//parmcell = parm*parm;		// number of interior points to be added
			// for now, we just add one node at cell center
				for(int inode = 0; inode < nnode[iel]; inode++)
				{
					c_x += coords(inpoel(iel,inode),0);
					c_y += coords(inpoel(iel,inode),1);
				}
				c_x /= nnode[iel];
				c_y /= nnode[iel];
				q.coords(numpoin+iel,0) = c_x;
				q.coords(numpoin+iel,1) = c_y;
				q.inpoel(iel,q.nnode[iel]-1) = numpoin+iel;
		}
	}
	std::cout << "UMesh: convertLinearToQuadratic(): Done." << std::endl;
	//q.inpoel.mprint();
	return q;
}

/**	Converts all quadrilaterals in a linear mesh into triangles, 
 * and returns the fully triangular mesh.
 */
template <typename scalar, int ndim>
UMesh<scalar,ndim> UMesh<scalar,ndim>::convertQuadToTri() const
{
	UMesh<scalar,ndim> tm;
	std::vector<std::vector<int>> elms;
	std::vector<std::vector<int>> volregs;
	std::vector<int> vr(ndtag, -1);
	int nnodet = 3;
	std::vector<int> element(nnodet,-1);
	int nelem2 = 0;

	for(int ielem = 0; ielem < nelem; ielem++)
	{
		if(nnode[ielem] == 4)
		{
			element[0] = inpoel.get(ielem,0);
			element[1] = inpoel.get(ielem,1);
			element[2] = inpoel.get(ielem,3);
			elms.push_back(element);
			
			for(int i = 0; i < ndtag; i++)
				vr[i] = vol_regions.get(ielem,i);
			volregs.push_back(vr);

			element[0] = inpoel.get(ielem,1);
			element[1] = inpoel.get(ielem,2);
			element[2] = inpoel.get(ielem,3);
			elms.push_back(element);
			volregs.push_back(vr);

			nelem2 += 2;
		}
		else if(nnode[ielem] == nnodet)
		{
			for(int i = 0; i < nnodet; i++)
				element[i] = inpoel.get(ielem,i);
			elms.push_back(element);
			
			for(int i = 0; i < ndtag; i++)
				vr[i] = vol_regions.get(ielem,i);
			volregs.push_back(vr);
			
			nelem2++;
		}
	}

	tm.nelem = nelem2;
	tm.npoin = npoin;
	tm.nface = nface;
	tm.nbtag = nbtag;
	tm.ndtag = ndtag;
	tm.nnofa = nnofa;

	tm.nnode.resize(nelem2);
	tm.nfael.resize(nelem2);

	tm.coords = coords;
	tm.inpoel.resize(tm.nelem, nnodet);
	tm.vol_regions.resize(tm.nelem, ndtag);
	tm.bface = bface;

	for(int ielem = 0; ielem < nelem2; ielem++)
	{
		for(int inode = 0; inode < nnodet; inode++)
			tm.inpoel(ielem, inode) = elms[ielem][inode];
		tm.nnode[ielem] = nnodet;
		tm.nfael[ielem] = nnodet;		// For linear elements
		for(int i = 0; i < ndtag; i++)
			tm.vol_regions(ielem,i) = volregs[ielem][i];
	}
	
	return tm;
}

template <typename scalar, int ndim>
void UMesh<scalar,ndim>::compute_elementsSurroundingPoints()
{
	//std::cout << "UMesh2d: compute_topological(): Elements surrounding points\n";
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
#ifdef DEBUG
	std::cout << "UMesh: compute_topological(): Elements surrounding elements...\n";
#endif
	//amat::Array2d<int> lpoin(npoin,1);
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
void UMesh<scalar,ndim>::compute_faceConnectivity()
{
#ifdef DEBUG
	std::cout << "UMesh: compute_topological(): Computing intfac..." << std::endl;
#endif
	nbface = naface = 0;
	// first run: calculate nbface
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nfael[ie]; in++)
		{
			int je = esuel(ie,in);
			if(je == -1)
			{
				//esuel(ie,in) = nelem+nbface;
				nbface++;
			}
		}
	}
	std::cout << "UMesh: compute_topological(): Number of boundary faces = " 
		<< nbface << std::endl;
	// calculate number of internal faces
	naface = nbface;
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nfael[ie]; in++)
		{
			int je = esuel(ie,in);
			if(je > ie && je < nelem) naface++;
		}
	}
	std::cout << "UMesh: compute_topological(): Number of all faces = " << naface << std::endl;

	//allocate intfac and elemface
	intfac.resize(naface,nnofa+2);
	elemface.resize(nelem,maxnfael);

	//reset face totals
	nbface = naface = 0;

	int in1, je, jnode;

	//second run: populate intfac
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nnode[ie]; in++)
		{
			je = esuel(ie,in);
			if(je == -1)
			{
				in1 = (in+1)%nnode[ie];
				esuel(ie,in) = nelem+nbface;
				intfac(nbface,0) = ie;
				intfac(nbface,1) = nelem+nbface;
				intfac(nbface,2) = inpoel(ie,in);
				intfac(nbface,3) = inpoel(ie,in1);
				elemface(ie,in) = nbface;

				nbface++;
			}
		}
	}
	naface = nbface;
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nnode[ie]; in++)
		{
			je = esuel(ie,in);
			if(je > ie && je < nelem)
			{
				in1 = (in+1)%nnode[ie];
				intfac(naface,0) = ie;
				intfac(naface,1) = je;
				intfac(naface,2) = inpoel.get(ie,in);
				intfac(naface,3) = inpoel.get(ie,in1);

				elemface(ie,in) = naface;
				for(jnode = 0; jnode < nnode[je]; jnode++)
					if(inpoel.get(ie,in1) == inpoel.get(je,jnode))
						elemface(je,jnode) = naface;

				naface++;
			}
		}
	}
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
	for(int ip = 0; ip < npoin; ip++)
	{
		lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
		// Loop over elements surrounding this point
		for(int ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
		{
			int ielem = esup(ie,0);		// element number

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
	for(int i = 0; i < npoin; i++)
		lpoin(i,0) = -1;
	for(int ip = 0; ip < npoin; ip++)
	{
		lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
		// Loop over elements surrounding this point
		for(int ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
		{
			int ielem = esup(ie,0);		// element number

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

template class UMesh<a_real>;

} // end namespace

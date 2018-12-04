/** \file
 * \brief Implementation of mesh readers
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
#include "utilities/aerrorhandling.hpp"
#include "utilities/mpiutils.hpp"

#include "meshreaders.hpp"

namespace fvens {

/// Reads mesh from Gmsh 2 format file
static MeshData readGmsh2(const std::string mfile);
static MeshData readSU2(const std::string mfile);

MeshData readMesh(const std::string mfile)
{
	// const int mpirank = get_mpi_rank(MPI_COMM_WORLD);
	MeshData mdata;

	std::vector<std::string> parts;
	boost::split(parts, mfile, boost::is_any_of("."));

	if(parts[parts.size()-1] == "su2")
		mdata = readSU2(mfile);
	else
		mdata = readGmsh2(mfile);

	// broadcast global mesh size data to all ranks
	// a_int numdata[] = {mdata.npoin, mdata.nelem, mdata.nface, mdata.maxnnode, mdata.maxnfael,
	//                    mdata.nnofa, mdata.nbtag, mdata.ndtag};
	// MPI_Bcast((void*)numdata, 8, FVENS_MPI_INT, 0, MPI_COMM_WORLD);
	// if(mpirank != 0) {
	// 	mdata.npoin = numdata[0];
	// 	mdata.nelem = numdata[1];
	// 	mdata.nface = numdata[2];
	// 	mdata.maxnnode = (int)numdata[3];
	// 	mdata.maxnfael = (int)numdata[4];
	// 	mdata.nnofa = (int)numdata[5];
	// 	mdata.nbtag = (int)numdata[6];
	// 	mdata.ndtag = (int)numdata[7];
	// }

	return mdata;
}

MeshData readGmsh2(const std::string mfile)
{
	MeshData m;

	int dum; double dummy; std::string dums; char ch;

	std::ifstream infile;
	open_file_toRead(mfile, infile);

	for(int i = 0; i < 4; i++)		//skip 4 lines
		do
			ch = infile.get();
		while(ch != '\n');

	infile >> m.npoin;
	m.coords.resize(m.npoin,NDIM);

	// read m.coords of points
	for(int i = 0; i < m.npoin; i++)
	{
		infile >> dum;
		for(int j = 0; j < NDIM; j++)
			infile >> m.coords(i,j);
		if(NDIM < 3) infile >> dummy;
	}
	infile >> dums;		// get 'endnodes'
	infile >> dums;		// get 'elements'

	int width_elms = 25;
	int nelm, elmtype, nbtags, ntags;
	/// elmtype is the standard element type in the Gmsh 2 mesh format - of either faces or elements
	m.ndtag = 0; m.nbtag = 0;
	infile >> nelm;
	amat::Array2d<a_int > elms(nelm,width_elms);
	m.nface = 0; m.nelem = 0;
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
				m.nnofa = 2;
				infile >> nbtags;
				if(nbtags > m.nbtag) m.nbtag = nbtags;
				for(int j = 0; j < nbtags; j++)
					infile >> elms(i,j+m.nnofa);		// get tags
				for(int j = 0; j < m.nnofa; j++)
					infile >> elms(i,j);			// get node numbers
				m.nface++;
				break;
			case(8): // quadratic edge
				m.nnofa = 3;
				infile >> nbtags;
				if(nbtags > m.nbtag) m.nbtag = nbtags;
				for(int j = 0; j < nbtags; j++)
					infile >> elms(i,j+m.nnofa);		// get tags
				for(int j = 0; j < m.nnofa; j++)
					infile >> elms(i,j);			// get node numbers
				m.nface++;
				break;
			case(2): // linear triangles
				nnodes[i] = 3;
				nfaels[i] = 3;
				m.nnofa = 2;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
				break;
			case(3):	// linear quads
				nnodes[i] = 4;
				nfaels[i] = 4;
				m.nnofa = 2;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
				break;
			case(9):	// quadratic triangles
				nnodes[i] = 6;
				nfaels[i] = 3;
				m.nnofa = 3;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
				break;
			case(16):	// quadratic quad (8 nodes)
				nnodes[i] = 8;
				nfaels[i] = 4;
				m.nnofa = 3;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
				break;
			case(10):	// quadratic quad (9 nodes)
				nnodes[i] = 9;
				nfaels[i] = 4;
				m.nnofa = 3;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
				break;
			default:
				std::cout << "! UMesh2d: readGmsh2(): Element type not recognized.";
				std::cout << " Setting as linear triangle." << std::endl;
				nnodes[i] = 3;
				nfaels[i] = 3;
				m.nnofa = 2;
				infile >> ntags;
				if(ntags > m.ndtag) m.ndtag = ntags;
				for(int j = 0; j < ntags; j++)
					infile >> elms(i,j+nnodes[i]);		// get tags
				for(int j = 0; j < nnodes[i]; j++)
					infile >> elms(i,j);			// get node numbers
				m.nelem++;
		}
	}
	/*std::cout << "UMesh2d: readGmsh2(): Done reading elms" << std::endl;
	for(int i = 0; i < nelm; i++)
		std::cout << nnodes[i] << " " << nfaels[i] << std::endl;*/

	m.nnode.reserve(m.nelem);
	m.nfael.reserve(m.nelem);

	// calculate max nnode and nfael
	m.maxnnode = nnodes[m.nface]; m.maxnfael = nfaels[m.nface];
	for(int i = 0; i < nelm; i++)
	{
		if(nnodes[i] > m.maxnnode)
			m.maxnnode = nnodes[i];
		if(nfaels[i] > m.maxnfael)
			m.maxnfael = nfaels[i];
	}

	if(m.nface > 0)
		m.bface.resize(m.nface, m.nnofa+m.nbtag);
	else std::cout << "readGmsh2(): WARNING: There is no boundary data!" << std::endl;

	m.inpoel.resize(m.nelem, m.maxnnode);
	m.vol_regions.resize(m.nelem, m.ndtag);

	std::cout << "readGmsh2(): No. of points: " << m.npoin
		<< ", number of elements: " << m.nelem
		<< ",\nnumber of boundary faces " << m.nface
		<< ", max no. of nodes per element: " << m.maxnnode
		<< ",\nno. of nodes per face: " << m.nnofa
		<< ", max faces per element: " << m.maxnfael << std::endl;

	// write into inpoel and bface
	// the first nface rows to be read are boundary faces
	for(int i = 0; i < m.nface; i++)
	{
		for(int j = 0; j < m.nnofa; j++)
			// -1 to correct for the fact that our numbering starts from zero
			m.bface(i,j) = elms(i,j)-1;
		for(int j = m.nnofa; j < m.nnofa+m.nbtag; j++)
			m.bface(i,j) = elms(i,j);
	}
	for(int i = 0; i < m.nelem; i++)
	{
		for(int j = 0; j < nnodes[i+m.nface]; j++)
			m.inpoel(i,j) = elms(i+m.nface,j)-1;
		for(int j = nnodes[i+m.nface]; j < m.maxnnode; j++)
			m.inpoel(i,j) = -1;
		for(int j = 0; j < m.ndtag; j++)
			m.vol_regions(i,j) = elms(i+m.nface,j+nnodes[i+m.nface]);
		m.nnode.push_back(nnodes[i+m.nface]);
		m.nfael.push_back(nfaels[i+m.nface]);
	}
	infile.close();

	return m;
}

MeshData readSU2(const std::string mfile)
{
	MeshData m;

	std::string dum;
	std::ifstream fin;
	open_file_toRead(mfile, fin);

	std::getline(fin, dum, '='); std::getline(fin,dum);
	int ndim = std::stoi(dum);
	if(ndim != NDIM)
		std::cout << "readSU2: Mesh is not " << NDIM << "-dimensional!\n";

	// read element node connectivity

	std::getline(fin, dum, '='); std::getline(fin,dum);
	m.nelem = std::stoi(dum);
	std::cout << "readSU2: Number of elements = " << m.nelem << std::endl;

	// Let's just assume a hybrid grid with triangles and quads
	m.maxnnode = 4; m.maxnfael = 4;
	m.inpoel.resize(m.nelem,m.maxnnode);
	m.nnode.resize(m.nelem); m.nfael.resize(m.nelem);

	for(a_int iel = 0; iel < m.nelem; iel++)
	{
		int id, ddum;
		fin >> id;
		switch(id)
		{
			case 5: // triangle
				m.nnode[iel] = 3;
				m.nfael[iel] = 3;
				break;
			case 9: // quad
				m.nnode[iel] = 4;
				m.nfael[iel] = 4;
				break;
			default:
				std::cout << "readSU2: Unknown element type!!\n";
		}

		for(int i = 0; i < m.nnode[iel]; i++)
			fin >> m.inpoel(iel,i);
		fin >> ddum;
	}
	// clear the newline
	std::getline(fin,dum);

	// read coordinates of nodes

	std::getline(fin, dum, '='); std::getline(fin,dum);
	m.npoin = std::stoi(dum);
#ifdef DEBUG
	std::cout << "readSU2: Number of points = " << m.npoin << std::endl;
#endif

	m.coords.resize(m.npoin,NDIM);
	for(a_int ip = 0; ip < m.npoin; ip++)
	{
		int ddum;
		for(int j = 0; j < NDIM; j++)
			fin >> m.coords(ip,j);
		fin >> ddum;
	}
	// clear the newline
	std::getline(fin,dum);

	// read boundary face data

	m.nbtag = 1; m.ndtag = 0;

	std::getline(fin, dum, '='); std::getline(fin,dum);
	int nbmarkers = std::stoi(dum);
#ifdef DEBUG
	std::cout << "UMesh2dh: readSU2: Number of BC markers = " << nbmarkers << std::endl;
#endif

	std::vector<std::vector<std::vector<a_int>>> bfacs(nbmarkers);

	std::vector<int> tags(nbmarkers);
	std::vector<a_int> numfacs(nbmarkers);
	m.nface = 0;

	for(int ib = 0; ib < nbmarkers; ib++)
	{
		std::getline(fin, dum, '='); std::getline(fin,dum);
		tags[ib] = std::stoi(dum);
		std::getline(fin, dum, '='); std::getline(fin,dum);
		numfacs[ib] = std::stoi(dum);
		m.nface += numfacs[ib];

		bfacs[ib].resize(numfacs[ib]);
		m.nnofa = 2;

		for(a_int iface = 0; iface < numfacs[ib]; iface++)
		{
			bfacs[ib][iface].resize(m.nnofa);
			int ddum;
			fin >> ddum;
			for(int inofa = 0; inofa < m.nnofa; inofa++)
				fin >> bfacs[ib][iface][inofa];
		}

		//clear newline, except the last one because it may not exist
		if(ib < nbmarkers-1)
			std::getline(fin,dum);
	}

	fin.close();

	std::cout << "UMesh2dh: readSU2: Number of boundary faces = " << m.nface << std::endl;

	m.bface.resize(m.nface,m.nnofa+m.nbtag);
	a_int count=0;
	for(int ib = 0; ib < nbmarkers; ib++)
	{
		for(a_int iface = 0; iface < numfacs[ib]; iface++)
		{
			for(int inofa = 0; inofa < m.nnofa; inofa++)
				m.bface(count,inofa) = bfacs[ib][iface][inofa];
			m.bface(count,m.nnofa) = tags[ib];

			count++;
		}
	}

	return m;
}

}

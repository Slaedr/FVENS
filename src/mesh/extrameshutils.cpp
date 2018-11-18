/** \file
 * \brief Implementation of some extra mesh utilities
 */

#include "extrameshutils.hpp"

namespace fvens {

/** (Deprecated) Reads the RDGFlo 'domn' format.
   NOTE: Make sure nnofa is mentioned after ndim and ntype in the mesh file.
   ntype makes no sense for us now.
   Can only be used for a linear mesh having all cells of the same shape.
*/
template <typename scalar>
void readDomn(const std::string mfile, UMesh2dh<scalar>& m)
{
	std::ifstream infile(mfile);
	if(!infile) {
		std::cout << "UMesh2dh: Could not open domn mesh file!\n";
		std::abort();
	}

	int nnode2, nfael2, ndim;

	// Do file handling here to populate npoin and nelem
	std::cout << "UMesh2dh: Reading domn mesh file...\n";
	char ch = '\0'; int dum = 0; double dummy;

	infile >> dum;
	infile >> ch;
	for(int i = 0; i < 4; i++)		//skip 4 lines
		do
			ch = infile.get();
		while(ch != '\n');
	infile >> ndim;
	infile >> nnode2;
	infile >> nfael2;
	infile >> nnofa;
	infile >> ch;			//get the newline
	do
		ch = infile.get();
	while(ch != '\n');
	infile >> nelem; infile >> npoin; infile >> nface;
	infile >> dummy; 				// get time
	ch = infile.get();			// clear newline

	nnode.resize(nelem,-1);
	nfael.resize(nelem,-1);

	nbtag = 2;
	ndtag = 2;

	//std::cout << "\nUTriMesh: Allocating coords..";
	coords.resize(npoin, NDIM);
	// temporary array to hold connectivity matrix
	amat::Array2d<a_int > elms(nelem,nnode2);
	//std::cout << "UTriMesh: Allocating bface...\n";
	bface.resize(nface, nnofa + nbtag);

	//std::cout << "UTriMesh: Allocation done.";

	do
		ch = infile.get();
	while(ch != '\n');

	//now populate inpoel
	for(int i = 0; i < nelem; i++)
	{
		//infile >> nnode[i];
		infile >> dum;
		nnode[i] = nnode2;
		//nfael[i] = nnode[i];		// NOTE: assuming linear element
		nfael[i] = nfael2;

		for(int j = 0; j < nnode[i]; j++)
			infile >> elms(i,j);

		do
			ch = infile.get();
		while(ch != '\n');
	}
	std::cout << "UMesh2dh: Populated inpoel.\n";

	maxnnode = 3;
	for(int i = 0; i < nelem; i++)
		if(nnode[i] > maxnnode)
			maxnnode = nnode[i];

	inpoel.resize(nelem, maxnnode);

	for(int i = 0; i < nelem; i++)
		for(int j = 0; j < nnode[i]; j++)
			inpoel(i,j) = elms.get(i,j);

	//Correct inpoel:
	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nnode[i]; j++)
			inpoel(i,j)--;
	}

	ch = infile.get();
	do
		ch = infile.get();
	while(ch != '\n');

	// populate coords
	for(int i = 0; i < npoin; i++)
	{
		infile >> dum;
		for(int j = 0; j < NDIM; j++)
			infile >> coords(i,j);
	}
	std::cout << "UMesh2dh: Populated coords.\n";

	// skip initial conditions
	ch = infile.get();
	for(int i = 0; i < npoin+2; i++)
	{
		do
			ch = infile.get();
		while(ch != '\n');
	}

	// populate bface
	for(int i = 0; i < nface; i++)
	{
		infile >> dum;
		for(int j = 0; j < NDIM + nbtag; j++)
		{
			infile >> bface(i,j);
		}
		if (i==nface-1) break;
		do
			ch = infile.get();
		while(ch!='\n');
	}
	std::cout << "UMesh2dh: Populated bface. Done reading mesh.\n";
	//correct first 2 columns of bface
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < 2; j++)
			bface(i,j)--;

	infile.close();

	vol_regions.resize(nelem, ndtag);
	vol_regions.zeros();

	std::cout << "UMesh2dh: Number of elements: " << nelem << ", number of points: " << npoin
		<< ", max number of nodes per element: " << maxnnode << std::endl;
	std::cout << "Number of boundary faces: " << nface << ", Number of dimensions: " << NDIM
		<< std::endl;

	// set flag_bpoin
	flag_bpoin.resize(npoin,1);
	flag_bpoin.zeros();
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < nnofa; j++)
			flag_bpoin(bface(i,j)) = 1;
}

/**	Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
 * NOTE: Make sure to execute [compute_topological()](@ref compute_topological)
 * before calling this function.
 */
template <typename scalar>
UMesh2dh<scalar> convertLinearToQuadratic(const UMesh2dh<scalar>& m)
{
	std::cout << "UMesh2d: convertLinearToQuadratic(): Producing quadratic mesh from linear mesh"
		<< std::endl;
	UMesh2dh<scalar> q;
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

	q.coords.resize(q.npoin, NDIM);
	q.inpoel.resize(q.nelem, q.maxnnode);
	q.bface.resize(q.nface, q.nnofa+q.nbtag);

	/// Next, we copy over low-order mesh data to the new mesh.
	for(int i = 0; i < npoin; i++)
		for(int j = 0; j < NDIM; j++)
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

		for(int idim = 0; idim < NDIM; idim++)
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

		for(int idim = 0; idim < NDIM; idim++)
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
		scalar c_x = 0, c_y = 0;

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
	std::cout << "UMesh2dh: convertLinearToQuadratic(): Done." << std::endl;
	//q.inpoel.mprint();
	return q;
}

/**	Converts all quadrilaterals in a linear mesh into triangles,
 * and returns the fully triangular mesh.
 */
template <typename scalar>
UMesh2dh<scalar> convertQuadToTri(const UMesh2dh<scalar>& m) const
{
	UMesh2dh<scalar> tm;
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

}

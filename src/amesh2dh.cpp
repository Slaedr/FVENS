#include "amesh2dh.hpp"

namespace acfd {

UMesh2dh::UMesh2dh() { 
	alloc_jacobians = false;
	alloc_lambda = false;
	neleminlambda = 3;
}

UMesh2dh::UMesh2dh(const UMesh2dh& other)
{
	npoin = other.npoin;
	nelem = other.nelem;
	nface = other.nface;
	ndim = other.ndim;
	nnode = other.nnode;
	naface = other.naface;
	nbface = other.nbface;
	nfael = other.nfael;
	maxnnode = other.maxnnode;
	maxnfael = other.maxnfael;
	nnofa = other.nnofa;
	nbtag = other.nbtag;
	ndtag = other.ndtag;
	nbpoin = other.nbpoin;
	coords = other.coords;
	inpoel = other.inpoel;
	bface = other.bface;
	vol_regions = other.vol_regions;
	esup = other.esup;
	esup_p = other.esup_p;
	psup = other.psup;
	psup_p = other.psup_p;
	esuel = other.esuel;
	intfac = other.intfac;
	intfacbtags = other.intfacbtags;
	bpoints = other.bpoints;
	bpointsb = other.bpointsb;
	bfacebp = other.bfacebp;
	bifmap = other.bifmap;
	ifbmap = other.ifbmap;
	isBoundaryMaps = other.isBoundaryMaps;
	//gallfa = other.gallfa;
	alloc_jacobians = other.alloc_jacobians;
	jacobians = other.jacobians;
}

UMesh2dh& UMesh2dh::operator=(const UMesh2dh& other)
{
	npoin = other.npoin;
	nelem = other.nelem;
	nface = other.nface;
	ndim = other.ndim;
	nnode = other.nnode;
	naface = other.naface;
	nbface = other.nbface;
	nfael = other.nfael;
	maxnnode = other.maxnnode;
	maxnfael = other.maxnfael;
	nnofa = other.nnofa;
	nbtag = other.nbtag;
	ndtag = other.ndtag;
	nbpoin = other.nbpoin;
	coords = other.coords;
	inpoel = other.inpoel;
	bface = other.bface;
	vol_regions = other.vol_regions;
	esup = other.esup;
	esup_p = other.esup_p;
	psup = other.psup;
	psup_p = other.psup_p;
	esuel = other.esuel;
	intfac = other.intfac;
	intfacbtags = other.intfacbtags;
	bpoints = other.bpoints;
	bpointsb = other.bpointsb;
	bfacebp = other.bfacebp;
	bifmap = other.bifmap;
	ifbmap = other.ifbmap;
	isBoundaryMaps = other.isBoundaryMaps;
	//gallfa = other.gallfa;
	alloc_jacobians = other.alloc_jacobians;
	jacobians = other.jacobians;
	return *this;
}

UMesh2dh::~UMesh2dh()
{
	if(alloc_lambda)
		delete [] lambda;
}

/** Reads Professor Luo's mesh file, which I call the 'domn' format.
   NOTE: Make sure nnofa is mentioned after ndim and ntype in the mesh file. ntype makes no sense for us now.
   Can only be used for linear mesh for now.
   MODIFIED FOR READING NON-HYBRID MESH FOR NOW.
*/
void UMesh2dh::readDomn(std::string mfile)
{
	std::ifstream infile(mfile);

	int nnode2, nfael2;
	
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
	coords.setup(npoin, ndim);
	// temporary array to hold connectivity matrix
	amat::Matrix<acfd_int > elms(nelem,nnode2);
	//std::cout << "UTriMesh: Allocating bface...\n";
	bface.setup(nface, nnofa + nbtag);
	
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
	
	inpoel.setup(nelem, maxnnode);

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
		for(int j = 0; j < ndim; j++)
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
		for(int j = 0; j < ndim + nbtag; j++)
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

	vol_regions.setup(nelem, ndtag);
	vol_regions.zeros();
	
	std::cout << "UMesh2dh: Number of elements: " << nelem << ", number of points: " << npoin << ", max number of nodes per element: " << maxnnode << std::endl;
	std::cout << "Number of boundary faces: " << nface << ", Number of dimensions: " << ndim << std::endl;

	/*if (nnode == 3) nmtens = 1;
	else if(nnode == 4) nmtens = 4;*/
	
	// set flag_bpoin
	flag_bpoin.setup(npoin,1);
	flag_bpoin.zeros();
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < nnofa; j++)
			flag_bpoin(bface(i,j)) = 1;
}

/// Reads mesh from Gmsh 2 format file
void UMesh2dh::readGmsh2(std::string mfile, int dimensions)
{
	std::cout << "UMesh2d: readGmsh2(): Reading mesh file...\n";
	int dum; double dummy; std::string dums; char ch;
	ndim = dimensions;

	std::ifstream infile(mfile);
	for(int i = 0; i < 4; i++)		//skip 4 lines
		do
			ch = infile.get();
		while(ch != '\n');

	infile >> npoin;
	std::cout << "UMesh2d: readGmsh2(): No. of points = " << npoin << std::endl;
	coords.setup(npoin,ndim);

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
	amat::Matrix<int > elms(nelm,width_elms);
	nface = 0; nelem = 0;
	std::vector<int> nnodes(nelm,0);
	std::vector<int> nfaels(nelm,0);
	//std::cout << "UMesh2d: readGmsh2(): Total number of elms is " << nelm << std::endl;

	for(int i = 0; i < nelm; i++)
	{
		infile >> dum;
		infile >> elmtype;
		/// elmtype is different for all faces and for all elements. However, meshes in which high-order and linear elements are both present are not supported.
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
				std::cout << "! UMesh2d: readGmsh2(): Element type not recognized. Setting as linear triangle." << std::endl;
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
		bface.setup(nface, nnofa+nbtag);
	else std::cout << "UMesh2d: readGmsh2(): NOTE: There is no boundary data!" << std::endl;

	inpoel.setup(nelem, maxnnode);
	vol_regions.setup(nelem, ndtag);

	std::cout << "UMesh2dh: readGmsh2(): Done. No. of points: " << npoin << ", number of elements: " << nelem << ", number of boundary faces " << nface << ",\n max number of nodes per element: " << maxnnode << ", number of nodes per face: " << nnofa << ", max number of faces per element: " << maxnfael << std::endl;

	// write into inpoel and bface
	// the first nface rows to be read are boundary faces
	for(int i = 0; i < nface; i++)
	{
		for(int j = 0; j < nnofa; j++)
			bface(i,j) = elms(i,j)-1;			// -1 to correct for the fact that our numbering starts from zero
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

	/*for(int i = 0; i < nelem; i++) 
	{	
		if (nnode == 3) nmtens[i] = 1;
		else if(nnode == 4) nmtens[i] = 4;
	}*/
	
	// set flag_bpoin
	flag_bpoin.setup(npoin,1);
	flag_bpoin.zeros();
	for(int i = 0; i < nface; i++)
		for(int j = 0; j < nnofa; j++)
			flag_bpoin(bface(i,j)) = 1;
}

/**	Stores (in array bpointsb) for each boundary point: the associated global point number and the two bfaces associated with it.
 * Also calculates bfacebp, which is like inpoel for boundary faces - it gives the boundary node number (according to bpointsb) of each local node of a bface.
 * \note Only for linear meshes.
 */
void UMesh2dh::compute_boundary_points()
{
	std::cout << "UMesh2dh: compute_boundary_points(): Calculating bpointsb structure"<< std::endl;

	// first, get number of boundary points

	nbpoin = 0;
	amat::Matrix<int > flagb(npoin,1);
	flagb.zeros();
	for(int iface = 0; iface < nface; iface++)
	{
		for(int inofa = 0; inofa < nnofa; inofa++)
			flagb(bface(iface,inofa)) = 1;
	}
	for(int ipoin = 0; ipoin < npoin; ipoin++)
		nbpoin += flagb(ipoin);

	std::cout << "UMesh2dh: compute_boundary_points(): No. of boundary points = " << nbpoin << std::endl;

	bpointsb.setup(nbpoin,3);
	for(int i = 0; i < nbpoin; i++)
		for(int j = 0; j < 3; j++)
			bpointsb(i,j) = -1;

	bfacebp.setup(nface,nnofa);
	
	amat::Matrix<double > lpoin(npoin,1);

	int bp = 0;

	// Next, populate bpointsb by iterating over faces. Also populate bfacebp, which holds the boundary points numbers of the 2 points in a bface.
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
			if(ibp==-1) std::cout << "UMesh2dh: compute_boundary_points(): Point not found!" << std::endl;
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
			if(ibp==-1) std::cout << "UMesh2d: compute_boundary_points(): Point not found!" << std::endl;
			bpointsb(ibp,1) = iface;
			bfacebp(iface,1) = ibp;
		}
	}
}

void UMesh2dh::printmeshstats()
{
	std::cout << "UMesh2dh: No. of points: " << npoin << ", number of elements: " << nelem << ", number of boundary faces " << nface << ", max number of nodes per element: " << maxnnode << ", number of nodes per face: " << nnofa << ", max number of faces per element: " << maxnfael << std::endl;
}

void UMesh2dh::writeGmsh2(std::string mfile)
{
	std::cout << "UMesh2dh: writeGmsh2(): writing mesh to file " << mfile << std::endl;
	// decide element type first, based on nfael/nnode and nnofa
	int elm_type = 2;
	int face_type = 1;

	if(nnofa == 3)
		face_type = 8;

	std::ofstream outf(mfile);
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

	//std::cout << "boundary faces\n";
	outf << "$Elements\n" << nelem+nface << '\n';
	// boundary faces first
	for(int iface = 0; iface < nface; iface++)
	{
		outf << iface+1 << " " << face_type << " " << nbtag;
		for(int i = nnofa; i < nnofa+nbtag; i++)
			outf << " " << bface(iface,i);			// write tags
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
		outf << nface+iel+1 << " " << elm_type << " " << ndtag;
		for(int i = 0; i < ndtag; i++)
			outf << " " << vol_regions(iel,i);
		for(int i = 0; i < nnode[iel]; i++)
			outf << " " << inpoel(iel,i)+1;
		outf << '\n';
	}
	outf << "$EndElements\n";

	outf.close();
}

/** \brief Computes area of linear triangular elements. So it can't be used for hybrid meshes.
 * 
 * \todo TODO: Generalize so that it works for quadrilateral meshes also
*/
void UMesh2dh::compute_jacobians()
{
	if(maxnnode == 3 || maxnnode == 4)
	{
		if (alloc_jacobians == false)
		{
			jacobians.setup(nelem, 1);
			alloc_jacobians = true;
		}

		for(int i = 0; i < gnelem(); i++)
		{
			jacobians(i,0) = gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) - gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)-gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) - gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1);

		}
	}
	else {
		std::cout << "UMesh2d: compute_jacobians(): ! Mesh is not linear. Cannot compute jacobians." << std::endl;
	}
}

// computes areas of linear triangles and quads
void UMesh2dh::compute_areas()
{
	area.setup(nelem,1);
	for(int i = 0; i < nelem; i++)
	{
		if(nnode[i] == 3)
			area(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) - gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)-gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) - gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
		else if(nnode[i]==4)
		{
			area(i,0) = 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) - gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)-gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) - gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1));
			area(i,0) += 0.5*(gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,2),1) - gcoords(ginpoel(i,3),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,2),0)-gcoords(ginpoel(i,3),0)) + gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,3),1) - gcoords(ginpoel(i,3),0)*gcoords(ginpoel(i,2),1));
		}
	}
}

void UMesh2dh::detect_negative_jacobians(std::ofstream& out)
{
	bool flagj = false;
	int nneg = 0;
	for(int i = 0; i < nelem; i++)
	{
		if(jacobians(i,0) <= 1e-15) {
			out << i << " " << jacobians(i,0) << '\n';
			flagj = true;
			nneg++;
		}
	}
	if(flagj == true) std::cout << "UMesh2d: detect_negative_jacobians(): There exist " << nneg << " element(s) with negative jacobian!!\n";
}

/// \todo: TODO: There is an issue with psup for some boundary nodes belonging to elements of different types. Correct this.
void UMesh2dh::compute_topological()
{

	std::cout << "UMesh2dh: compute_topological(): Calculating and storing topological information...\n";
	/// 1. Elements surrounding points
	//std::cout << "UMesh2d: compute_topological(): Elements surrounding points\n";
	esup_p.setup(npoin+1,1);
	esup_p.zeros();

	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nnode[i]; j++)
		{
			esup_p(inpoel(i,j)+1,0) += 1;	// inpoel(i,j) + 1 : the + 1 is there because the storage corresponding to the first node begins at 0, not at 1
		}
	}
	// Now make the members of esup_p cumulative
	for(int i = 1; i < npoin+1; i++)
		esup_p(i,0) += esup_p(i-1,0);
	// Now populate esup
	esup.setup(esup_p(npoin,0),1);
	esup.zeros();
	for(int i = 0; i < nelem; i++)
	{
		for(int j = 0; j < nnode[i]; j++)
		{
			int ipoin = inpoel(i,j);
			esup(esup_p(ipoin,0),0) = i;		// now put that element no. in the space pointed to by esup_p(ipoin)
			esup_p(ipoin,0) += 1;				// an element corresponding to ipoin has been found - increment esup_p for that point
		}
	}
	//But now esup_p holds increased values - each member increased by the number elements surrounding the corresponding point.
	// So correct this.
	for(int i = npoin; i >= 1; i--)
		esup_p(i,0) = esup_p(i-1,0);
	esup_p(0,0) = 0;
	// Elements surrounding points is now done.

	/// 2. Points surrounding points
	std::cout << "UMesh2dh: compute_topological(): Points surrounding points\n";
	psup_p.setup(npoin+1,1);
	psup_p.zeros();
	psup_p(0,0) = 0;
	amat::Matrix<int > lpoin(npoin,1);  // The ith member indicates the global point number of which the ith point is a surrounding point
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
			int inode;
			for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				if(inpoel(ielem,jnode) == ip) inode = jnode;

			std::vector<bool> nbd(nnode[ielem]);		// contains true if that local node number is connected to a particular local node.
			for(int j = 0; j < nnode[ielem]; j++)
				nbd[j] = false;

			if(nnode[ielem] == 3)
				for(int i = 0; i < nbd.size(); i++)
					nbd[i] = true;
			else if(nnode[ielem] == 4)
				for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				{
					if(jnode == perm(0,nnode[ielem]-1,inode,1) || jnode == perm(0,nnode[ielem]-1, inode, -1))
						nbd[jnode] = true;
				}

			//loop over nodes of the element
			for(int inode = 0; inode < nnode[ielem]; inode++)
			{
				//Get global index of this node
				int jpoin = inpoel(ielem, inode);
				if(lpoin(jpoin,0) != ip && nbd[inode])		// test if this point as already been counted as a surrounding point of ip, and whether it's connected to ip.
				{
					istor++;
					lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
				}
			}
		}
		psup_p(ip+1,0) = istor;
	}

	psup.setup(istor,1);
	//std::cout << "+++ " << istor << std::endl;

	//second pass: populate psup
	istor = 0;
	for(int i = 0; i < npoin; i++) lpoin(i,0) = -1;	// initialize lpoin to -1
	for(int ip = 0; ip < npoin; ip++)
	{
		lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
		// Loop over elements surrounding this point
		for(int ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
		{
			int ielem = esup(ie,0);		// element number

			// find local node number of ip in ielem
			int inode;
			for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				if(inpoel(ielem,jnode) == ip) inode = jnode;

			std::vector<bool> nbd(nnode[ielem]);		// nbd[j] contains true if ip is connected to local node number j of ielem.
			for(int j = 0; j < nnode[ielem]; j++)
				nbd[j] = false;

			if(nnode[ielem] == 3)
				for(int i = 0; i < nbd.size(); i++)
					nbd[i] = true;
			else if(nnode[ielem] == 4)
				for(int jnode = 0; jnode < nnode[ielem]; jnode++)
				{
					if(jnode == perm(0,nnode[ielem]-1,inode,1) || jnode == perm(0,nnode[ielem]-1, inode, -1))
						nbd[jnode] = true;
				}

			//loop over nodes of the element
			for(int inode = 0; inode < nnode[ielem]; inode++)
			{
				//Get global index of this node
				int jpoin = inpoel(ielem, inode);
				if(lpoin(jpoin,0) != ip && nbd[inode])		// test of this point as already been counted as a surrounding point of ip
				{
					psup(istor,0) = jpoin;
					istor++;
					lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
				}
			}
		}
	}
	//Points surrounding points is now done.

	/// 3. Elements surrounding elements
	std::cout << "UMesh2dh: compute_topological(): Elements surrounding elements...\n";

	//amat::Matrix<int> lpoin(npoin,1);
	esuel.setup(nelem, maxnfael);
	for(int ii = 0; ii < nelem; ii++)
		for(int jj = 0; jj < maxnfael; jj++)
			esuel(ii,jj) = -1;
	//lpofa.mprint();
	amat::Matrix<int > lhelp(nnofa,1);
	lhelp.zeros();
	lpoin.zeros();

	for(int ielem = 0; ielem < nelem; ielem++)
	{
		// first get lpofa for this element
		amat::Matrix<int > lpofai(nfael[ielem], nnofa);	// lpofa(i,j) holds local node number of jth node of ith face (j in [0:nnofa], i in [0:nfael])
		amat::Matrix<int > lpofaj;							// to be initialized for each jelem
		for(int i = 0; i < nfael[ielem]; i++)
		{
			for(int j = 0; j < nnofa; j++)
			{
				lpofai(i,j) = (i+j) % nnode[ielem];		// fine as long as operands of % are not negative
			}
		}

		for(int ifael = 0; ifael < nfael[ielem]; ifael++)
		{
			for(int i = 0; i < nnofa; i++)
			{
				lhelp(i,0) = inpoel(ielem, lpofai(ifael,i));	// lhelp stores global node nos. of current face of current element
				lpoin(lhelp(i,0)) = 1;
			}
			int ipoin = lhelp(0);
			for(int istor = esup_p(ipoin); istor < esup_p(ipoin+1); istor++)
			{
				int jelem = esup(istor);

				if(jelem != ielem)
				{
					// setup lpofa for jelem
					lpofaj.setup(nfael[jelem],nnofa);
					for(int i = 0; i < nfael[jelem]; i++)
						for(int j = 0; j < nnofa; j++)
							lpofaj(i,j) = (i+j)%nnode[jelem];

					for(int jfael = 0; jfael < nfael[jelem]; jfael++)
					{
						//Assume that no. of nodes in face ifael is same as that in face jfael
						int icoun = 0;
						for(int jnofa = 0; jnofa < nnofa; jnofa++)
						{
							int jpoin = inpoel(jelem, lpofaj(jfael,jnofa));
							if(lpoin(jpoin)==1) icoun++;
						}
						if(icoun == nnofa)		// nnofa is 2
						{
							esuel(ielem,ifael) = jelem;
							esuel(jelem,jfael) = ielem;
						}
					}
				}
			}
			for(int i = 0; i < nnofa; i++)
				lpoin(lhelp(i)) = 0;
		}
	}

	/** Computes, for each face, the elements on either side, the starting node and the ending node of the face. This is stored in intfac. 
	 * Also computes unit normals to, and lengths of, each face as well as boundary flags of boundary faces, in gallfa.
	 * The orientation of the face is such that the element with smaller index is always to the left of the face, while the element with greater index is always to the right of the face.
	 * Also computes element-face connectivity array elemface in the same loop which computes intfac.
	 * \note After the following portion, esuel holds (nelem + face no.) for each ghost cell, instead of -1 as before.
	 */

	std::cout << "UMesh2dh: compute_topological(): Computing intfac..." << std::endl;
	nbface = naface = 0;
	// first run: calculate nbface
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nnode[ie]; in++)
		{
			int je = esuel(ie,in);
			if(je == -1)
			{
				//esuel(ie,in) = nelem+nbface;
				nbface++;
			}
		}
	}
	std::cout << "UMesh2dh: compute_topological(): Number of boundary faces = " << nbface << std::endl;
	// calculate number of internal faces
	naface = nbface;
	for(int ie = 0; ie < nelem; ie++)
	{
		for(int in = 0; in < nnode[ie]; in++)
		{
			int je = esuel(ie,in);
			if(je > ie && je < nelem) naface++;
		}
	}
	std::cout << "UMesh2dh: compute_topological(): Number of all faces = " << naface << std::endl;

	//allocate intfac and elemface
	intfac.setup(naface,nnofa+2);
	elemface.setup(nelem,maxnfael);

	//reset face totals
	nbface = naface = 0;

	int in1, je, jnode, jlocnode;

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

	/// Finally, calculates bpoints.

	//first get number of bpoints
	nbpoin = 0;
	amat::Matrix<int > isbpflag(npoin,1);
	isbpflag.zeros();
	for(int i = 0; i < nface; i++)
	{
		for(int j = 0; j < nnofa; j++)
			isbpflag(bface(i,j)) = 1;
	}
	for(int i = 0; i < npoin; i++)
		if(isbpflag(i)==1) nbpoin++;

	std::cout << "UMesh2dh: compute_topological(): Number of boundary points = " << nbpoin << std::endl;

	//Allocate bpoints
	/*bpoints.setup(nbpoin,3);		// We need 1 field for global point number and in 2D linear meshes, we need 2 more for surrounding faces
	for(int i = 0; i < nbpoin; i++)
		for(int j = 0; j < 3; j++)
			bpoints(i,j) = -1;

	int bp = 0;

	// Next, populate bpoints by iterating over intfac faces
	lpoin.zeros();		// lpoin will be 1 if the point has been visited
	for(int iface = 0; iface < nbface; iface++)
	{
		int p1, p2;
		p1 = intfac.get(iface,2+0);
		p2 = intfac.get(iface,2+1);
		std::cout << p1 << "," << p2 << " ";

		if(lpoin.get(p1) == 0)	// if this point has not been visited before
		{
			bpoints(bp,0) = p1;
			bpoints(bp,2) = iface;
			bp++;
			lpoin(p1) = 1;
		}
		else
		{
			// search bpoints for point p1
			int ibp=-1;
			for(int i = 0; i < bp; i++)
			{
				if(bpoints.get(i,0) == p1) ibp = i;
			}

			bpoints(ibp,2) = iface;
		}

		if(lpoin.get(p2) == 0)	// if this point has not been visited before
		{
			bpoints(bp,0) = p2;
			bpoints(bp,1) = iface;
			bp++;
			lpoin(p2) = 1;
		}
		else
		{
			// search bpoints for point p2
			int ibp;
			for(int i = 0; i < bp; i++)
			{
				if(bpoints.get(i,0) == p2) ibp = i;
			}

			bpoints(ibp,1) = iface;
		}
	}*/
	std::cout << "UMesh2dh: compute_topological(): Done." << std::endl;
}

/** Assumption: order of nodes of boundary faces is such that normal points outside, when normal is calculated as
 * 		nx = y2 - y1, ny = -(x2-x1).
 */
void UMesh2dh::compute_face_data()
{
	int i, j, p1, p2;

	//Now compute normals and lengths (only linear meshes!)
	gallfa.setup(naface, 3+nbtag);
	for(i = 0; i < naface; i++)
	{
		gallfa(i,0) = coords(intfac(i,3),1) - coords(intfac(i,2),1);
		gallfa(i,1) = -1.0*(coords(intfac(i,3),0) - coords(intfac(i,2),0));
		gallfa(i,2) = sqrt(pow(gallfa(i,0),2) + pow(gallfa(i,1),2));
		//Normalize the normal vector components
		gallfa(i,0) /= gallfa(i,2);
		gallfa(i,1) /= gallfa(i,2);
	}

	//Populate boundary flags in gallfa
	std::cout << "UTriMesh: compute_face_data(): Storing boundary flags in gallfa...\n";
	for(int ied = 0; ied < nbface; ied++)
	{
		p1 = intfac(ied,2);
		p2 = intfac(ied,3);
		
		if(nbface != nface) { 
			std::cout << "UMesh2dh: Calculation of number of boundary faces is wrong!" << std::endl; 
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
						gallfa(ied,3+j) = bface.get(i,nnofa+j);
						gallfa(ied,3+j) = bface.get(i,nnofa+j);
					}
				}
			}
		}
	}

	std::cout << "UTriMesh: compute_face_data(): Done.\n";
}

void UMesh2dh::compute_boundary_maps()
{
	// iterate over bfaces and find corresponding intfac face for each bface
	bifmap.setup(nbface,1);
	ifbmap.setup(nbface,1);

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
						inter[j] = true;			// if jth node of ibface has a node of iface, it belongs to iface; set the corresp. boolean to true
						break;
					}
			}

			/*for(int i = 0; i < nnofa; i++)
				std::cout << inter[i];
			std::cout << std::endl;*/

			for(int b = 0; b < nnofa; b++)
				if(inter[b] == false) final1 = false;						// if any node of ibface failed to find a node of iface, ibface is not the same as iface

			if(final1 == true) inface = iface;
		}

		if(inface != -1) {
			bifmap(inface) = ibface;
			ifbmap(ibface) = inface;
		}
		else {
			std::cout << "UMesh2d: compute_boundary_maps(): ! intfac face corresponding to " << ibface << "th bface not found!!" << std::endl;
			continue;
		}
	}
	isBoundaryMaps = true;
}

void UMesh2dh::writeBoundaryMapsToFile(std::string mapfile)
{
	if(isBoundaryMaps == false) {
		std::cout << "UMesh2d: writeBoundaryMapsToFile(): ! Boundary maps not available!" << std::endl;
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

void UMesh2dh::readBoundaryMapsFromFile(std::string mapfile)
{
	std::ifstream ofile(mapfile);
	std::string dum; int sz;
	ofile >> sz >>  dum;
	std::cout << "UMesh2d: readBoundaryMapsFromFile(): Number of boundary faces in file = " << sz << std::endl;
	bifmap.setup(sz,1);
	ifbmap.setup(sz,1);

	for(int i = 0; i < nbface; i++)
		ofile >> bifmap(i);

	ofile >> dum;
	for(int i = 0; i < nbface; i++)
		ofile >> ifbmap(i);

	ofile.close();
	isBoundaryMaps = true;
}

void UMesh2dh::compute_intfacbtags()
{
	/// Populate intfacbtags with boundary markers of corresponding bfaces

	intfacbtags.setup(nbpoin,nbtag);

	if(isBoundaryMaps == false)
	{
		std::cout << "UMesh2d: compute_intfacbtags(): ! Boundary maps are not available!" << std::endl;
		return;
	}

	for(int ibface = 0; ibface < nface; ibface++)
	{
		for(int j = 0; j < nbtag; j++)
			intfacbtags(ifbmap(ibface),j) = bface(ibface,nnofa+j);
	}
}

/**	Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
	NOTE: Make sure to execute [compute_topological()](@ref compute_topological) before calling this function.
*/
UMesh2dh UMesh2dh::convertLinearToQuadratic()
{
	std::cout << "UMesh2d: convertLinearToQuadratic(): Producing quadratic mesh from linear mesh" << std::endl;
	UMesh2dh q;
	if(nnofa != 2) { std::cout << "! UMesh2d: convertLinearToQuadratic(): Mesh is not linear!!" << std::endl; return q;}

	int parm = 1;		// 1 extra node per face
	
	/// We first calculate: total number of non-simplicial elements; nnode, nfael in each element; mmax nnode and max nfael.
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

	q.ndim = ndim;
	q.npoin = npoin + naface + nelemnonsimp;
	q.nelem = nelem;
	q.nface = nface;
	q.nbface = nbface;
	q.naface = naface;
	q.nnofa = nnofa+parm;
	q.nbtag = nbtag;
	q.ndtag = ndtag;

	q.coords.setup(q.npoin, q.ndim);
	q.inpoel.setup(q.nelem, q.maxnnode);
	q.bface.setup(q.nface, q.nnofa+q.nbtag);

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

	int ied, p1, p2, ielem, jelem, idim, inode, lp1, lp2, ifa;

	/// We then iterate over faces, introducing the required number of points in each face.
	
	//std::cout << "UMesh2d: convertLinearToQuadratic(): Iterating over boundary faces..." << std::endl;
	// iterate over boundary faces
	for(ied = 0; ied < nbface; ied++)
	{
		ielem = intfac(ied,0);
		jelem = intfac(ied,1);
		p1 = intfac(ied,2);
		p2 = intfac(ied,3);

		for(idim = 0; idim < ndim; idim++)
			q.coords(npoin+ied*parm,idim) = (coords(p1,idim) + coords(p2,idim))/2.0;

		for(inode = 0; inode < nnode[ielem]; inode++)
		{
			if(p1 == inpoel(ielem,inode)) lp1 = inode;
			if(p2 == inpoel(ielem,inode)) lp2 = inode;
		}

		// in the left element, the new point is in face ip1 (ie, the face whose first point is ip1 in CCW order)
		q.inpoel(ielem, nnode[ielem]+lp1) = npoin+ied*parm;

		// find the bface that this face corresponds to
		for(ifa = 0; ifa < nface; ifa++)
		{
			if((p1 == bface(ifa,0) && p2 == bface(ifa,1)) || (p1 == bface(ifa,1) && p2 == bface(ifa,0)))	// face found
			{
				q.bface(ifa,nnofa) = npoin+ied*parm;
			}
		}
	}

	//std::cout << "UMesh2d: convertLinearToQuadratic(): Iterating over internal faces..." << std::endl;
	// iterate over internal faces
	for(ied = nbface; ied < naface; ied++)
	{
		ielem = intfac(ied,0);
		jelem = intfac(ied,1);
		p1 = intfac(ied,2);
		p2 = intfac(ied,3);

		for(idim = 0; idim < ndim; idim++)
			q.coords(npoin+ied*parm,idim) = (coords(p1,idim) + coords(p2,idim))/2.0;

		// First look at left element
		for(inode = 0; inode < nnode[ielem]; inode++)
		{
			if(p1 == inpoel(ielem,inode)) lp1 = inode;
			if(p2 == inpoel(ielem,inode)) lp2 = inode;
		}

		// in the left element, the new point is in face ip1 (ie, the face whose first point is ip1 in CCW order)
		q.inpoel(ielem, nnode[ielem]+lp1) = npoin+ied*parm;

		// Then look at right element
		for(inode = 0; inode < nnode[jelem]; inode++)
		{
			if(p1 == inpoel(jelem,inode)) lp1 = inode;
			if(p2 == inpoel(jelem,inode)) lp2 = inode;
		}

		// in the right element, the new point is in face ip2
		q.inpoel(jelem, nnode[jelem]+lp2) = npoin+ied*parm;
	}
	
	// for non-simplicial mesh, add extra points at cell-centres as well

	int parmcell;
	int numpoin = npoin+naface*parm;		// next global point number to be added
	// get cell centres
	for(int iel = 0; iel < nelem; iel++)
	{
		parmcell = 1;		// number of extra nodes per cell in the interior of the cell
		double c_x = 0, c_y = 0;

		if(nnode[iel] == 4)	
		{
			parmcell = parm*parm;		// number of interior points to be added
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

/**	Converts all quadrilaterals in a linear mesh into triangles, and returns the fully triangular mesh.
*/
UMesh2dh UMesh2dh::convertQuadToTri() const
{
	UMesh2dh tm;
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
	tm.ndim = ndim;
	tm.npoin = npoin;
	tm.nface = nface;
	tm.nbtag = nbtag;
	tm.ndtag = ndtag;
	tm.nnofa = nnofa;

	tm.nnode.resize(nelem2);
	tm.nfael.resize(nelem2);

	tm.coords = coords;
	tm.inpoel.setup(tm.nelem, nnodet);
	tm.vol_regions.setup(tm.nelem, ndtag);
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

} // end namespace

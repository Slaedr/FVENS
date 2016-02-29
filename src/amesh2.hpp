/** @file amesh2.hpp
 * @brief Data structure and setup for 2D unstructured triangular mesh.
 * @author Aditya Kashi
 * @date Feb 5, 2015
 * 
 * Feb 26, 2015: Adding mesh-movement function movemesh2d_bs().
 */

#ifndef _GLIBCXX_IOSTREAM
#include <iostream>
#endif
#ifndef _GLIBCXX_FSTREAM
#include <fstream>
#endif
#ifndef _GLIBCXX_STRING
#include <string>
#endif
#ifndef _GLIBCXX_CMATH
#include <cmath>
#endif
#ifdef _OPENMP
#ifndef OMP_H
#include <omp.h>
#endif
#endif

#ifndef __AMATRIX2_H
#include "amatrix2.hpp"
#endif

#ifndef __ADATASTRUCTURES_H
#include "adatastructures.hpp"
#endif

#define __AMESH2D_H

using namespace std;
using namespace amat;

namespace acfd {

const int n_extra_fields_in_bface = 2;

/// Unstructured triangular mesh class
class UTriMesh
{
private:
	int npoin;
	int nelem;
	int nface;
	int ndim;
	
	/// number of nodes to an element
	int nnode;
	/// number of faces to an element (equal to number of edges to an element in 2D)
	int nfael;
	/// number of node in a face -- needs to be generalized in case of general grids
	int nnofa;
	/// total number of (internal and boundary) faces
	int naface;
	/// number of boundary faces as calculated by compute_face_data(), as opposed to nface which is read from file
	int nbface;
	/// number of boundary points !! not used !!
	int nbpoin;
	
	Matrix<double> coords;
	Matrix<int> inpoel;
	Matrix<int> bface;

	// The following 4 are for moving mesh
	/// for the basicspring mesh mpvement, and for output of interior node data
	Matrix<int> bflag;		
	Matrix<int> bflag2;
	Matrix<double> cosa;
	Matrix<double> sina;
	
public:
	/// elements surrounding point
	Matrix<int> esup;
	/// stores positions corresponding to points in esup
	Matrix<int> esup_p;
	Matrix<int> psup;
	Matrix<int> psup_p;
	Matrix<int> esuel;

	/// Face data structure (topological)
	/** naface x 4 matrix: holds for each face: 
	 * 1. index of left element, 
	 * 2. index of right element, 
	 * 3. index of starting point of face, and 
	 * 4. index of ending point of face
	 */
	Matrix<int> intfac;
	
	/// Face data structure (geometrical)
	/** naface x 3 matrix: holds for each face: 
	 * 1. x-component of normal 
	 * 2. y-component of normal 
	 * 3. Measure (length) of face. 
	 * Note that normal points from left element towards right element (refer to intfac)
	 */
	Matrix<double> gallfa;

	bool alloc_jacobians;
	Matrix<double> jacobians;

public:

	UTriMesh() {}

	UTriMesh(Matrix<double>* co, Matrix<int>* inp, int n_poin, int n_elem, int n_node, int n_nofa)
	{
		alloc_jacobians = false;
		coords = *co;
		inpoel = *inp;
		npoin = n_poin;
		nelem = n_elem;
		nnode = n_node;
		nnofa = n_nofa;
	}
	/* UTriMesh()
	{
		npoin = 0; nelem = 0; nface = 0; ndim = 0; nnode = 0;
		coords.setup(1,1,ROWMAJOR);
		inpoel.setup(1,1,ROWMAJOR);
		bface.setup(1,1,ROWMAJOR);
	} */

	/// Reads mesh from the argument file stream referring to a domn file, and computes some topological data structures.
	/** Computes elements surrounding points, points surrounding points and elements surrounding elements.
	 */
	UTriMesh(ifstream& infile)
	{
		alloc_jacobians = false;

		// Do file handling here to populate npoin and nelem
		cout << "UTriMesh: Reading mesh file...\n";
		char ch = '\0'; int dum = 0; double dummy;

		//infile >> ch;
		infile >> dum;
		infile >> ch;
		for(int i = 0; i < 4; i++)		//skip 4 lines
			do
				ch = infile.get();
			while(ch != '\n');
		infile >> ndim;
		infile >> nnode;
		infile >> ch;			//get the newline
		do
			ch = infile.get();
		while(ch != '\n');
		infile >> nelem; infile >> npoin; infile >> nface;
		infile >> dummy; 				// get time
		ch = infile.get();			// clear newline

		cout << "UTriMesh: Number of elements: " << nelem << ", number of points: " << npoin << ", number of nodes per element: " << nnode << endl;
		cout << "Number of boundary faces: " << nface << ", Number of dimensions: " << ndim;

		//cout << "\nUTriMesh: Allocating coords..";
		coords.setup(npoin, ndim, ROWMAJOR);
		//cout << "\nUTriMesh: Allocating inpoel..\n";
		inpoel.setup(nelem, nnode, ROWMAJOR);
		//cout << "UTriMesh: Allocating bface...\n";
		bface.setup(nface, ndim + n_extra_fields_in_bface, ROWMAJOR);

		nfael = 3;	// number of faces per element
		nnofa = 2;	// number of nodes per face

		//cout << "UTriMesh: Allocation done.";

		do
			ch = infile.get();
		while(ch != '\n');

		//now populate inpoel
		for(int i = 0; i < nelem; i++)
		{
			infile >> dum;

			for(int j = 0; j < nnode; j++)
				infile >> inpoel(i,j);

			do
				ch = infile.get();
			while(ch != '\n');
		}
		cout << "\nUTriMesh: Populated inpoel.";

		//Correct inpoel:
		for(int i = 0; i < nelem; i++)
		{
			for(int j = 0; j < nnode; j++)
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
		cout << "\nUTriMesh: Populated coords.\n";
		//coords.mprint();

		ch = infile.get();
		for(int i = 0; i < npoin+2; i++)
		{
			do
				ch = infile.get();
			while(ch != '\n');
		}

		for(int i = 0; i < nface; i++)
		{
			infile >> dum;
			for(int j = 0; j < ndim + n_extra_fields_in_bface; j++)
			{
				infile >> bface(i,j);
			}
			if (i==nface-1) break;
			do
				ch = infile.get();
			while(ch!='\n');
		}
		cout << "UTriMesh: Populated bface. Done reading mesh.\n";
		//correct first 2 columns of bface
		for(int i = 0; i < nface; i++)
			for(int j = 0; j < 2; j++)
				bface(i,j)--;

		//------------- Calculate other topological properties -----------------
		cout << "UTriMesh: Calculating and storing topological information...\n";
		//1. Elements surrounding points
		cout << "UTriMesh: Elements surrounding points\n";
		esup_p.setup(npoin+1,1,ROWMAJOR);
		esup_p.zeros();

		for(int i = 0; i < nelem; i++)
		{
			for(int j = 0; j < nnode; j++)
			{
				esup_p(inpoel(i,j)+1,0) += 1;	// inpoel(i,j) + 1 : the + 1 is there because the storage corresponding to the first node begins at 0, not at 1
			}
		}
		// Now make the members of esup_p cumulative
		for(int i = 1; i < npoin+1; i++)
			esup_p(i,0) += esup_p(i-1,0);
		// Now populate esup
		esup.setup(esup_p(npoin,0),1,ROWMAJOR);
		esup.zeros();
		for(int i = 0; i < nelem; i++)
		{
			for(int j = 0; j < nnode; j++)
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

		//2. Points surrounding points
		cout << "UTriMesh: Points surrounding points\n";
		psup_p.setup(npoin+1,1,ROWMAJOR);
		psup_p.zeros();
		psup_p(0,0) = 0;
		Matrix<int> lpoin(npoin,1);  // The ith member indicates the global point number of which the ith point is a surrounding point
		for(int i = 0; i < npoin; i++) lpoin(i,0) = -1;	// initialize this vector to -1
		int istor = 0;

		// first pass: calculate storage needed for psup
		for(int ip = 0; ip < npoin; ip++)
		{
			lpoin(ip,0) = ip;		// the point ip itself is not counted as a surrounding point of ip
			// Loop over elements surrounding this point
			for(int ie = esup_p(ip,0); ie <= esup_p(ip+1,0)-1; ie++)
			{
				int ielem = esup(ie,0);		// element number
				//loop over nodes of the element
				for(int inode = 0; inode < nnode; inode++)
				{
					//Get global index of this node
					int jpoin = inpoel(ielem, inode);
					if(lpoin(jpoin,0) != ip)		// test of this point as already been counted as a surrounding point of ip
					{
						istor++;
						//psup(istor,0) = jpoin;	// ! can't do this yet - psup not allocated!
						lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
					}
				}
			}
			psup_p(ip+1,0) = istor;
		}

		psup.setup(istor,1,ROWMAJOR);

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
				//loop over nodes of the element
				for(int inode = 0; inode < nnode; inode++)
				{
					//Get global index of this node
					int jpoin = inpoel(ielem, inode);
					if(lpoin(jpoin,0) != ip)		// test of this point as already been counted as a surrounding point of ip
					{
						psup(istor,0) = jpoin;
						istor++;
						lpoin(jpoin,0) = ip;		// set this point as a surrounding point of ip
					}
				}
			}
			//psup_p(ip+1,0) = istor;
		}
		//Points surrounding points is now done.

		// 3. Elements surrounding elements
		cout << "UTriMesh: Elements surrounding elements...\n";

		esuel.setup(nelem, nfael, ROWMAJOR);
		for(int ii = 0; ii < nelem; ii++)
			for(int jj = 0; jj < nfael; jj++)
				esuel(ii,jj) = -1;
		Matrix<int> lpofa(nfael, nnofa);	// lpofa(i,j) holds local node number of jth node of ith face (j in {0,1}, i in {0,1,2})
		lpofa(0,0) = 1; lpofa(0,1) = 2;
		lpofa(1,0) = 2; lpofa(1,1) = 0;
		lpofa(2,0) = 0; lpofa(2,1) = 1;
		Matrix<int> lhelp(nnofa,1);
		lhelp.zeros();
		lpoin.zeros();

		for(int ielem = 0; ielem < nelem; ielem++)
		{
			for(int ifael = 0; ifael < nfael; ifael++)
			{
				for(int i = 0; i < nnofa; i++)
				{
					lhelp(i,0) = inpoel(ielem, lpofa(ifael,i));	// lhelp stores global node nos. of current face of current element
					lpoin(lhelp(i,0)) = 1;
				}
				int ipoin = lhelp(0);
				for(int istor = esup_p(ipoin); istor < esup_p(ipoin+1); istor++)
				{
					int jelem = esup(istor);
					if(jelem != ielem)
					{
						for(int jfael = 0; jfael < nfael; jfael++)
						{
							//Assume that no. of nodes in face ifael is same as that in face jfael
							int icoun = 0;
							for(int jnofa = 0; jnofa < nnofa; jnofa++)
							{
								int jpoin = inpoel(jelem, lpofa(jfael,jnofa));
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
		//cout << "UTriMesh: Elements surrounding elements done.\n";

		//for moving mesh stuff:
		//bflag = bflags();
		//bflag2 = bflags2();
		cout << "UTriMesh: Done.\n";
	}

	UTriMesh(UTriMesh& other)
	{
		/*npoin = other.npoin;
		nelem = other.nelem;
		nface = other.nface;
		ndim = other.ndim;
		nnode = other.nnode;
		coords = other.coords;
		inpoel = other.inpoel;
		bface = other.bface; */

		npoin = other.npoin;
		nelem = other.nelem;
		nface = other.nface;
		ndim = other.ndim;
		nnode = other.nnode;
		naface = other.naface;
		nbface = other.nbface;
		nfael = other.nfael;
		nnofa = other.nnofa;
		coords = other.coords;
		inpoel = other.inpoel;
		bface = other.bface;
		esup = other.esup;
		esup_p = other.esup_p;
		psup = other.psup;
		psup_p = other.psup_p;
		esuel = other.esuel;
		intfac = other.intfac;
		gallfa = other.gallfa;
		alloc_jacobians = other.alloc_jacobians;
		jacobians = other.jacobians;
	}

	UTriMesh& operator=(UTriMesh& other)
	{
		npoin = other.npoin;
		nelem = other.nelem;
		nface = other.nface;
		ndim = other.ndim;
		nnode = other.nnode;
		naface = other.naface;
		nbface = other.nbface;
		nfael = other.nfael;
		nnofa = other.nnofa;
		coords = other.coords;
		inpoel = other.inpoel;
		bface = other.bface;
		esup = other.esup;
		esup_p = other.esup_p;
		psup = other.psup;
		psup_p = other.psup_p;
		esuel = other.esuel;
		intfac = other.intfac;
		gallfa = other.gallfa;
		alloc_jacobians = other.alloc_jacobians;
		jacobians = other.jacobians;
		return *this;
	}

	// Returns x or y coordinate (depending on dim) of node number pointno. Numbering of points begins
	// from 0. Numbering of dimensions begins from 0.
	double gcoords(int pointno, int dim) const
	{
		return coords.get(pointno,dim);
	}

	// Returns global node number of locnode th local node of element number elemno. Numberings for both
	// begins from 0
	int ginpoel(int elemno, int locnode) const
	{
		return inpoel.get(elemno, locnode);
	}

	int gbface(int faceno, int val) const
	{
		return bface.get(faceno, val);
	}

	void setcoords(Matrix<double>* c)
	{ coords = *c; }

	void setinpoel(Matrix<int>* inp)
	{ inpoel = *inp; }

	void setbface(Matrix<int>* bf)
	{ bface = *bf; }

	Matrix<double>* getcoords()
	{ return &coords; }

	int gesup(int i) const { return esup.get(i); }
	int gesup_p(int i) const { return esup_p.get(i); }
	int gpsup(int i) const { return psup.get(i); }
	int gpsup_p(int i) const { return psup_p.get(i); }
	/// returns element number at face opposite to node number jnode
	int gesuel(int ielem, int jnode) const { return esuel.get(ielem, jnode); }
	double gjacobians(int ielem) const { return jacobians.get(ielem,0); }

	int gnpoin() const { return npoin; }
	int gnelem() const { return nelem; }
	int gnface() const { return nface; }
	int gnbface() const { return nbface; }
	int gnnode() const { return nnode; }
	int gndim() const { return ndim; }
	int gnaface() const {return naface; }
	int gnfael() const { return nfael; }
	int gnnofa() const { return nnofa; }
	int gnbpoin() const { cout << "UTriMesh: ! Invalid access to 'gnbpoin()'!!" << endl; return nbpoin; }

	void compute_jacobians()
	{
		if (alloc_jacobians == false)
		{
			jacobians.setup(nelem, 1, ROWMAJOR);
			alloc_jacobians = true;
		}

		for(int i = 0; i < gnelem(); i++)
		{
			// geoel(i,0) = D(i) = a1*b2 - a2*b1 :

			jacobians(i,0) = gcoords(ginpoel(i,0),0)*(gcoords(ginpoel(i,1),1) - gcoords(ginpoel(i,2),1)) - gcoords(ginpoel(i,0),1)*(gcoords(ginpoel(i,1),0)-gcoords(ginpoel(i,2),0)) + gcoords(ginpoel(i,1),0)*gcoords(ginpoel(i,2),1) - gcoords(ginpoel(i,2),0)*gcoords(ginpoel(i,1),1);
		}
	}

	void detect_negative_jacobians(ofstream& out) const
	{
		bool flagj = false;
		for(int i = 0; i < nelem; i++)
		{
			if(jacobians.get(i,0) <= 1e-15) {
				out << i << " " << jacobians.get(i,0) << '\n';
				flagj = true;
			}
		}
		if(flagj == true) cout << "UTriMesh: detect_negative_jacobians(): There exist element(s) with negative jacobian!!\n";
	}

	/// Computes face data structures.
	/** Computes, for each face, 
	 * 1. the left element (with less element index)
	 * 2. the right element (with greater element index)
	 * 3. the starting node and 
	 * 4. the ending node of the face. 
	 * This is stored in intfac. Also computes unit normals to, and lengths of, each face and stores boundary flags of boundary faces, in gallfa.
	 * \note After this function, esuel holds (nelem + face no.) for each ghost cell, instead of -1 as before.
	 */
	void compute_face_data()
	{
		nbface = naface = 0;
		// first run: calculate nbface
		for(int ie = 0; ie < nelem; ie++)
		{
			for(int in = 0; in < nnode; in++)
			{
				//int in1 = perm(0,nnode-1,in,1);
				//int in2 = perm(0,nnode-1,in1,1);
				int je = esuel(ie,in);
				if(je == -1)
				{
					//esuel(ie,in) = nelem+nbface;
					nbface++;
				}
			}
		}
		cout << "UTriMesh: compute_face_data(): Number of boundary faces = " << nbface << endl;
		// calculate number of internal faces
		naface = nbface;
		for(int ie = 0; ie < nelem; ie++)
		{
			for(int in = 0; in < nnode; in++)
			{
				//int in1 = perm(0,nnode-1,in,1);
				//int in2 = perm(0,nnode-1,in1,1);
				int je = esuel(ie,in);
				if(je > ie && je < nelem) naface++;
			}
		}
		cout << "UTriMesh: compute_face_data(): Number of all faces = " << naface << endl;

		//allocate intfac
		intfac.setup(naface,4,ROWMAJOR);

		//reset face totals
		nbface = naface = 0;

		//second run: populate intfac
		int in, in1, in2, je;
		for(int ie = 0; ie < nelem; ie++)
		{
			for(in = 0; in < nnode; in++)
			{
				//in1 = perm(0,nnode-1,in,1);
				//in2 = perm(0,nnode-1,in1,1);
				in1 = (in+1) % nnode;
				in2 = (in1+1) % nnode;
				je = esuel(ie,in);
				if(je == -1)
				{
					esuel(ie,in) = nelem+nbface;
					intfac(nbface,0) = ie;
					intfac(nbface,1) = nelem+nbface;
					intfac(nbface,2) = inpoel(ie,in1);
					intfac(nbface,3) = inpoel(ie,in2);

					nbface++;
				}
			}
		}
		
		naface = nbface;
		for(int ie = 0; ie < nelem; ie++)
		{
			for(in = 0; in < nnode; in++)
			{
				//in1 = perm(0,nnode-1,in,1);
				//in2 = perm(0,nnode-1,in1,1);
				in1 = (in+1) % nnode;
				in2 = (in1+1) % nnode;
				je = esuel(ie,in);
				if(je > ie && je < nelem)
				{
					intfac(naface,0) = ie;
					intfac(naface,1) = je;
					intfac(naface,2) = inpoel(ie,in1);
					intfac(naface,3) = inpoel(ie,in2);
					naface++;
				}
			}
		}

		//Now compute normals and lengths
		gallfa.setup(naface, 3+n_extra_fields_in_bface, ROWMAJOR);
		for(int i = 0; i < naface; i++)
		{
			gallfa(i,0) = coords(intfac(i,3),1) - coords(intfac(i,2),1);
			gallfa(i,1) = -1.0*(coords(intfac(i,3),0) - coords(intfac(i,2),0));
			gallfa(i,2) = sqrt(pow(gallfa(i,0),2) + pow(gallfa(i,1),2));
			//Normalize the normal vector components
			gallfa(i,0) /= gallfa(i,2);
			gallfa(i,1) /= gallfa(i,2);
		}

		//Populate boundary flags in gallfa
		cout << "UTriMesh: compute_face_data(): Storing boundary flags in gallfa...\n";
		for(int ied = 0; ied < nbface; ied++)
		{
			int p1 = intfac(ied,2);
			int p2 = intfac(ied,3);
			// Assumption: order of nodes of boundary faces is such that normal points outside, when normal is calculated as
			//nx = y2 - y1, ny = -(x2-x1).
			if(nbface != nface) { cout << "UTriMesh: Calculation of number of boundary faces is wrong!\n"; break; }
			for(int i = 0; i < nface; i++)
			{
				if(bface(i,0) == p1 || bface(i,1) == p1)
				{
					if(bface(i,1) == p2 || bface(i,0) == p2)
					{
						gallfa(ied,3) = bface(i,2);
						gallfa(ied,4) = bface(i,3);
					}
				}
			}
		}

		cout << "UTriMesh: compute_face_data(): Done.\n";
	}

	void compute_lengths_and_normals()		// Also happening in compute_face_data
	{
		for(int i = 0; i < naface; i++)
		{
			gallfa(i,0) = coords(intfac(i,3),1) - coords(intfac(i,2),1);
			gallfa(i,1) = -1.0*(coords(intfac(i,3),0) - coords(intfac(i,2),0));
			gallfa(i,2) = sqrt(pow(gallfa(i,0),2) + pow(gallfa(i,1),2));
			//Normalize the normal vector components
			gallfa(i,0) /= gallfa(i,2);
			gallfa(i,1) /= gallfa(i,2);
		}
	}

	void allocate_edge_angles()		// call only after calling compute_face_data()
	{
		cosa.setup(naface,1);
		sina.setup(naface,1);
	}

	void compute_edge_angles()
	{
		cout << "UTriMesh: Computing edge angles...\n";
		// calculate cos(a) and sin(a) where 'a' is angle made by edge (face in 2D) with the x axis
		for(int ied = 0; ied < naface; ied++)
		{
			cosa(ied) = -gallfa(ied,1)/gallfa(ied,2);
			sina(ied) = gallfa(ied,0)/gallfa(ied,2);
		}
	}

	int gintfac(int face, int i) const { return intfac.get(face,i); }
	double ggallfa(int elem, int i) const {return gallfa.get(elem,i); }

	// Returns an array whose ith element is 1 if the ith node is a boundary node
	Matrix<int> bflags() const
	{
		Matrix<int> flags(npoin, 1);
		flags.zeros();
		int ip[2];

		for(int b = 0; b < nface; b++)
		{
			ip[0] = bface.get(b,0);
			ip[1] = bface.get(b,1);
			flags(ip[0],0) = 1;
			flags(ip[1],0) = 1;
		}

		return flags;
	}

	Matrix<double> return_boundary_points()
	{
		cout << "UTriMesh: return_boundary_points(): Separating boundary points\n";
		bflag = bflags();
		int k = 0;
		for(int i = 0; i < bflag.rows(); i++)
		{
			if(bflag(i) == 1) k++;
		}
		// k is now the number of boundary nodes
		cout << "UTriMesh: return_boundary_points(): No. boundary nodes is " << k << endl;

		Matrix<double> bnodes(k,ndim);
		int j = 0;
		for(int i = 0; i < bflag.rows(); i++)
		{
			if(bflag(i)==1){
			 	for(int dim = 0; dim < ndim; dim++)
					bnodes(j,dim) = coords(i,dim);
				j++;
			}
		}
		return bnodes;
	}

	Matrix<double> return_interior_points()
	{
		bflag = bflags();
		int k = 0;
		for(int i = 0; i < bflag.rows(); i++)
		{
			if(bflag(i) == 0) k++;
		}
		// k is now the number of interior nodes

		Matrix<double> intnodes(k,ndim);
		int j = 0;
		for(int i = 0; i < bflag.rows(); i++)
		{
			if(bflag(i)==0){
			 	for(int dim = 0; dim < ndim; dim++)
					intnodes(k,dim) = coords(i,dim);
				j++;
			}
		}
		return intnodes;
	}

	/// Writes mesh to file in Gmsh2 format
	void writeGmsh2(string mfile) const
	{
		ofstream outf(mfile);

		outf << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
		outf << "$Nodes\n" << npoin << '\n';
		for(int ip = 0; ip < npoin; ip++)
		{
			outf << ip+1 << " " << coords.get(ip,0) << " " << coords.get(ip,1) << " " << 0 << '\n';
		}
		outf << "$Elements\n" << nelem+nface << '\n';
		// boundary faces first
		for(int iface = 0; iface < nface; iface++)
		{
			outf << iface+1 << " 1 2 0 1";
			for(int i = 0; i < nnofa; i++)
				outf << " " << bface.get(iface,i)+1;
			outf << '\n';
		}
		for(int iel = 0; iel < nelem; iel++)
		{
			outf << nface+iel+1 << " 2 2 0 2";
			for(int i = 0; i < nnode; i++)
				outf << " " << inpoel.get(iel,i)+1;
			outf << '\n';
		}
		outf << "$EndElements\n";

		outf.close();
	}
};

} // end namespace acfd

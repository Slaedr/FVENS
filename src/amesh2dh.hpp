/** @file amesh2dh.hpp
 * @brief Contains a class to handle 2D hybrid meshes containing triangles and quadrangles
 * @author Aditya Kashi
 */

#ifndef __AMESH2DH_H
#define __AMESH2DH_H

#ifndef __ACONSTANTS_H
#include <aconstants.hpp>
#endif

#ifndef __AMATRIX_H
#include <amatrix.hpp>
#endif
#ifndef __ADATASTRUCTURES_H
#include <adatastructures.hpp>
#endif

namespace acfd {

/// General hybrid unstructured mesh class supporting triangular and quadrangular elements
class UMesh2dh
{
private:
	int npoin;					///< Number of nodes
	int nelem;					///< Number of elements
	int nface;					///< Number of boundary faces
	int ndim;					///< Dimension of the mesh
	std::vector<int> nnode;		///< number of nodes to an element, for each element
	int maxnnode;				///< Maximum number of nodes per element for any element
	std::vector<int> nfael;		///< number of faces to an element (equal to number of edges to an element in 2D) for each element
	int maxnfael;				///< Maximum number of faces per element for any element
	int nnofa;					///< number of nodes in a face
	int naface;					///< total number of (internal and boundary) faces
	int nbface;					///< number of boundary faces as calculated by compute_face_data(), as opposed to nface which is read from file
	int nbpoin;					///< number of boundary points
	int nbtag;					///< number of tags for each boundary face
	int ndtag;					///< number of tags for each element
	amat::Matrix<double > coords;				///< Specifies coordinates of each node
	amat::Matrix<int > inpoel;				///< Interconnectivity matrix: lists node numbers of nodes in each element
	amat::Matrix<int > bface;					///< Boundary face data: lists nodes belonging to a boundary face and contains boudnary markers
	amat::Matrix<double > vol_regions;		///< to hold volume region markers, if any
	amat::Matrix<acfd_real > flag_bpoin;		///< Holds 1 or 0 for each point depending on whether or not that point is a boundary point

	/// List of indices of [esup](@ref esup) corresponding to nodes
	amat::Matrix<int > esup_p;
	/// List of elements surrounding points. 
	/** Integers pointing to particular points' element lists are stored in [esup_p](@ref esup_p). */
	amat::Matrix<int > esup;
	/// Lists of indices of psup corresponding to nodes (points)
	amat::Matrix<int > psup_p;
	
	/// List of nodes surrounding nodes
	/** Integers pointing to particular nodes' node lists are stored in [psup_p](@ref psup_p)
	 */
	amat::Matrix<int > psup;
	
	/// Elements surrounding elements
	amat::Matrix<int > esuel;
	/// Face data structure - contains info about elements and nodes associated with a face
	amat::Matrix<int > intfac;
	/// Holds boundary tags (markers) corresponding to intfac
	amat::Matrix<int > intfacbtags;
	/// Holds face numbers of faces making up an element
	amat::Matrix<int> elemface;
	
	/** @brief Boundary points list
	 * 
	 * bpoints contains: bpoints(0) = global point number, bpoints(1) = first containing intfac face (face with intfac's second point as this point), 
	 * bpoints(2) = second containing intfac face (face with intfac's first point as this point)
	 */
	amat::Matrix<int > bpoints;
	
	/// Like bpoints, but stores bface numbers corresponding to each face, rather than intfac faces
	amat::Matrix<int > bpointsb;

	/// Stores boundary-points numbers (defined by bpointsb) of the two points making up a particular bface.
	amat::Matrix<int > bfacebp;

	amat::Matrix<int > bifmap;				///< relates boundary faces in intfac with bface, ie, bifmap(intfac no.) = bface no.
	amat::Matrix<int > ifbmap;				///< relates boundary faces in bface with intfac, ie, ifbmap(bface no.) = intfac no.
	bool isBoundaryMaps;			///< Specifies whether bface-intfac maps have been created

	bool alloc_jacobians;			///< Flag indicating whether space has been allocated for jacobians
	amat::Matrix<double > jacobians;		///< Contains jacobians of each (linear triangular) element
	amat::Matrix<acfd_real> area;			///< Contains area of each element (either triangle or quad)
	amat::Matrix<acfd_real> gallfa;			///< Stores lengths and normals for linear mesh faces
	
	/// Contains Knupp's node-local areas for each node of each element. 
	/** If the elements are triangles, it contains just 1 value for each element. If elements are quads, there are 4 values for each element, one associated with each node. */
	amat::Matrix<double > alpha;

	/// Contains Knupp's 3 coeffs of metric tensor for each node of each element. 
	/** In case of triangles, it just contains 3 coeffs for each element. In case of quads, we need to store 3 coeffs for each node of each element. */
	amat::Matrix<double >* lambda;

	std::vector<int> nmtens;				///< number of metric tensors required for each element - 1 for triangles and 4 for quads.
	int neleminlambda;				///< number of coeffs in lambda per element per node.
	bool alloc_lambda;				///< Contains true if alpha and lambda have been allocated.

public:
	UMesh2dh();
	UMesh2dh(const UMesh2dh& other);
	UMesh2dh& operator=(const UMesh2dh& other);
	~UMesh2dh();
		
	/* Functions to get mesh data. */

	double gcoords(int pointno, int dim) const
	{
		return coords.get(pointno,dim);
	}
	int ginpoel(int elemno, int locnode) const
	{
		return inpoel.get(elemno, locnode);
	}
	int gbface(int faceno, int val) const
	{
		return bface.get(faceno, val);
	}

	amat::Matrix<double >* getcoords()
	{ return &coords; }

	int gesup(int i) const { return esup.get(i); }
	int gesup_p(int i) const { return esup_p.get(i); }
	int gpsup(int i) const { return psup.get(i); }
	int gpsup_p(int i) const { return psup_p.get(i); }
	int gesuel(int ielem, int jnode) const { return esuel.get(ielem, jnode); }
	acfd_int gelemface(acfd_int ielem, int inode) const { return elemface.get(ielem,inode); }
	int gintfac(int face, int i) const { return intfac.get(face,i); }
	int gintfacbtags(int face, int i) const { return intfacbtags.get(face,i); }
	int gbpoints(int poin, int i) const { return bpoints.get(poin,i); }
	int gbpointsb(int poin, int i) const { return bpointsb.get(poin,i); }
	int gbfacebp(int iface, int i) const { return bfacebp.get(iface,i); }
	int gbifmap(int intfacno) const { return bifmap.get(intfacno); }
	int gifbmap(int bfaceno) const { return ifbmap.get(bfaceno); }
	double gjacobians(int ielem) const { return jacobians.get(ielem,0); }
	acfd_real garea(acfd_int ielem) const { return area.get(ielem,0); }
	acfd_real ggallfa(acfd_int iface, int index) const { return gallfa.get(iface,index); }
	int gflag_bpoin(const acfd_int pointno) const { return flag_bpoin.get(pointno); }

	int gnpoin() const { return npoin; }
	int gnelem() const { return nelem; }
	int gnface() const { return nface; }
	int gnbface() const { return nbface; }
	int gnnode(int ielem) const { return nnode[ielem]; }
	int gndim() const { return ndim; }
	int gnaface() const {return naface; }
	int gnfael(int ielem) const { return nfael[ielem]; }
	int gnnofa() const { return nnofa; }
	int gnbtag() const{ return nbtag; }
	int gndtag() const { return ndtag; }
	int gnbpoin() const { return nbpoin; }

	/* Functions to set some mesh data structures. */
	/// set coordinates of a certain point; 'set' counterpart of the 'get' function [gcoords](@ref gcoords).
	void scoords(const acfd_int pointno, const int dim, const acfd_real value)
	{
		coords(pointno,dim) = value;
	}

	void setcoords(amat::Matrix<double >* c)
	{ coords = *c; }

	void setinpoel(amat::Matrix<int >* inp)
	{ inpoel = *inp; }

	void setbface(amat::Matrix<int >* bf)
	{ bface = *bf; }

	void modify_bface_marker(int iface, int pos, int number)
	{ bface(iface, pos) = number; }

	/** \brief Reads Professor Luo's mesh file, which I call the 'domn' format.
	 * 
	 * \note NOTE: Make sure nfael and nnofa are mentioned after ndim and nnode in the mesh file.
	*/
	void readDomn(std::string mfile);

	/// Reads mesh from Gmsh 2 format file
	void readGmsh2(std::string mfile, int dimensions);
	
	/// Stores (in array bpointsb) for each boundary point: the associated global point number and the two bfaces associated with it.
	void compute_boundary_points();

	void printmeshstats();
	void writeGmsh2(std::string mfile);

	void compute_jacobians();
	void detect_negative_jacobians(std::ofstream& out);
	
	/// computes areas of linear triangles and quads
	void compute_areas();
	
	/** Computes data structures for 
	 * elements surrounding point (esup), 
	 * points surrounding point (psup), 
	 * elements surrounding elements (esuel), 
	 * elements surrounding faces along with points in faces (intfac),
	 * element-face connectivity array elemface (for each facet of each element, it stores the intfac face number)
	 * a list of boundary points with correspong global point numbers and containing boundary faces (according to intfac) (bpoints).
	 * \note
	 * - Use only after setup()
	 * - Currently only works for linear mesh
	 */
	void compute_topological();
	
	/// Computes normals and lengths, and sets boundary face tags for all faces in gallfa; only for linear meshes!
	/** \note Uses intfac, so call only after compute_topological, only for linear mesh
	 */
	void compute_face_data();

	/// Iterates over bfaces and finds the corresponding intfac face for each bface
	/** Stores this data in the boundary label maps [ifbmap](@ref ifbmap) and [bifmap](@ref bifmap).
	 */
	void compute_boundary_maps();
	
	/// Writes the boundary point maps [ifbmap](@ref ifbmap) and [bifmap](@ref bifmap) to a file
	void writeBoundaryMapsToFile(std::string mapfile);
	/// Reads the boundary point maps [ifbmap](@ref ifbmap) and [bifmap](@ref bifmap) from a file
	void readBoundaryMapsFromFile(std::string mapfile);
	
	/// Populate [intfacbtags](@ref intfacbtags) with boundary markers of corresponding bfaces
	void compute_intfacbtags();

	/**	\brief Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
	 * 
	 * \note Make sure to execute [compute_topological()](@ref compute_topological) before calling this function.
	*/
	UMesh2dh convertLinearToQuadratic();

	UMesh2dh convertQuadToTri() const;
};


} // end namespace
#endif

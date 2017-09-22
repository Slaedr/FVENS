/** @file amesh2dh.hpp
 * @brief Contains a class to handle 2D hybrid meshes containing triangles and quadrangles
 * @author Aditya Kashi
 */

#ifndef __AMESH2DH_H
#define __AMESH2DH_H

#ifndef __ACONSTANTS_H
#include "aconstants.hpp"
#endif

#ifndef __AARRAY2D_H
#include "aarray2d.hpp"
#endif

namespace acfd {

/// General hybrid unstructured mesh class supporting triangular and quadrangular elements
class UMesh2dh
{
private:
	a_int npoin;					///< Number of nodes
	a_int nelem;					///< Number of elements
	a_int nface;					///< Number of boundary faces
	int ndim;						///< \deprecated Dimension of the mesh
	std::vector<int> nnode;			///< number of nodes to an element, for each element
	int maxnnode;					///< Maximum number of nodes per element for any element
	std::vector<int> nfael;     ///< number of faces to an element 
	                            ///< (equal to number of edges to an element in 2D) for each element
	int maxnfael;					///< Maximum number of faces per element for any element
	int nnofa;						///< number of nodes in a face
	a_int naface;					///< total number of (internal and boundary) faces
	a_int nbface;                   ///< number of boundary faces as calculated
	a_int nbpoin;					///< number of boundary points
	int nbtag;						///< number of tags for each boundary face
	int ndtag;						///< number of tags for each element
	/// Coordinates of nodes
	amat::Array2d<double > coords;
	/// Interconnectivity matrix: lists node numbers of nodes in each element
	amat::Array2d<a_int > inpoel; 
	/// Boundary face data: lists nodes belonging to a boundary face and contains boudnary markers
	amat::Array2d<a_int > bface;	
	amat::Array2d<double > vol_regions;			///< to hold volume region markers, if any
	/// Holds 1 or 0 for each point depending on whether or not that point is a boundary point
	amat::Array2d<a_real > flag_bpoin;	

	/// List of indices of [esup](@ref esup) corresponding to nodes
	amat::Array2d<a_int > esup_p;
	/// List of elements surrounding points. 
	/** Integers pointing to particular points' element lists are stored in [esup_p](@ref esup_p).*/
	amat::Array2d<a_int > esup;
	/// Lists of indices of psup corresponding to nodes (points)
	amat::Array2d<a_int > psup_p;
	
	/// List of nodes surrounding nodes
	/** Integers pointing to particular nodes' node lists are stored in [psup_p](@ref psup_p)
	 */
	amat::Array2d<a_int > psup;
	
	/// Elements surrounding elements
	amat::Array2d<a_int > esuel;
	/// Face data structure - contains info about elements and nodes associated with a face
	amat::Array2d<a_int > intfac;
	/// Holds boundary tags (markers) corresponding to intfac
	amat::Array2d<int > intfacbtags;
	/// Holds face numbers of faces making up an element
	amat::Array2d<a_int> elemface;
	
	/** @brief Boundary points list
	 * 
	 * bpoints contains: bpoints(0) = global point number, 
	 * bpoints(1) = first containing intfac face (face with intfac's second point as this point), 
	 * bpoints(2) = second containing intfac face (face with intfac's first point as this point)
	 */
	amat::Array2d<int > bpoints;
	/// Like bpoints, but stores bface numbers corresponding to each face, rather than intfac faces
	amat::Array2d<int > bpointsb;
	/// Stores boundary-points numbers (defined by bpointsb) of the two points making up a bface
	amat::Array2d<int > bfacebp;
	/// Relates boundary faces in intfac with bface, ie, bifmap(intfac no.) = bface no.
	amat::Array2d<int > bifmap;
	/// Relates boundary faces in bface with intfac, ie, ifbmap(bface no.) = intfac no.
	amat::Array2d<int > ifbmap;
	bool isBoundaryMaps;			///< Specifies whether bface-intfac maps have been created
	/// Flag indicating whether space has been allocated for jacobians
	bool alloc_jacobians;			
	amat::Array2d<a_real> jacobians;	///< Contains jacobians of each (linear triangular) element
	
	amat::Array2d<a_real> area;			///< Contains area of each element (either triangle or quad)

	/// Stores lengths and unit normals for linear mesh faces
	/** For each face, the first two entries are x- and y-components of the unit normal,
	 * the third component is the length.
	 * The unit normal points towards the cell with greater index.
	 */
	amat::Array2d<a_real> gallfa;

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
	a_int ginpoel(int elemno, int locnode) const
	{
		return inpoel.get(elemno, locnode);
	}
	a_int gbface(int faceno, int val) const
	{
		return bface.get(faceno, val);
	}

	amat::Array2d<double >* getcoords()
	{ return &coords; }

	a_int gesup(a_int i) const { return esup.get(i); }
	a_int gesup_p(a_int i) const { return esup_p.get(i); }
	a_int gpsup(a_int i) const { return psup.get(i); }
	a_int gpsup_p(a_int i) const { return psup_p.get(i); }
	a_int gesuel(a_int ielem, int jnode) const { return esuel.get(ielem, jnode); }
	a_int gelemface(a_int ielem, int inode) const { return elemface.get(ielem,inode); }
	a_int gintfac(a_int face, int i) const { return intfac.get(face,i); }
	int gintfacbtags(a_int face, int i) const { return intfacbtags.get(face,i); }
	a_real garea(const a_int ielem) const { return area.get(ielem,0); }
	a_real ggallfa(a_int iface, int index) const { return gallfa.get(iface,index); }
	int gflag_bpoin(const a_int pointno) const { return flag_bpoin.get(pointno); }
	/*int gbpoints(a_int poin, int i) const { return bpoints.get(poin,i); }
	int gbpointsb(a_int poin, int i) const { return bpointsb.get(poin,i); }
	int gbfacebp(a_int iface, int i) const { return bfacebp.get(iface,i); }
	int gbifmap(a_int intfacno) const { return bifmap.get(intfacno); }
	int gifbmap(a_int bfaceno) const { return ifbmap.get(bfaceno); }
	double gjacobians(a_int ielem) const { return jacobians.get(ielem,0); }*/

	a_int gnpoin() const { return npoin; }
	a_int gnelem() const { return nelem; }
	a_int gnface() const { return nface; }
	a_int gnbface() const { return nbface; }
	int gnnode(int ielem) const { return nnode[ielem]; }
	int gndim() const { return ndim; }
	a_int gnaface() const {return naface; }
	int gnfael(int ielem) const { return nfael[ielem]; }
	int gnnofa() const { return nnofa; }
	int gnbtag() const{ return nbtag; }
	int gndtag() const { return ndtag; }
	//int gnbpoin() const { return nbpoin; }

	/// Set coordinates of a certain point
	/** 'set' counterpart of the 'get' function [gcoords](@ref gcoords).
	 */
	void scoords(const a_int pointno, const int dim, const a_real value)
	{
		coords(pointno,dim) = value;
	}

	void setcoords(amat::Array2d<double >* c)
	{ coords = *c; }

	void setinpoel(amat::Array2d<int >* inp)
	{ inpoel = *inp; }

	void setbface(amat::Array2d<int >* bf)
	{ bface = *bf; }

	void modify_bface_marker(int iface, int pos, int number)
	{ bface(iface, pos) = number; }

	/// Reads mesh from Gmsh 2 format file
	void readGmsh2(std::string mfile, int dimensions);

	/** \brief Reads 'domn' format
	 * 
	 * \note NOTE: Make sure nfael and nnofa are mentioned after ndim and nnode in the mesh file.
	 * \deprecated Please use Gmsh format instead.
	*/
	void readDomn(std::string mfile);
	
	/** Stores (in array bpointsb) for each boundary point: the associated global point number 
	 * and the two bfaces associated with it.
	 */
	void compute_boundary_points();

	void printmeshstats();
	void writeGmsh2(std::string mfile);

	void compute_jacobians();
	void detect_negative_jacobians(std::ofstream& out);
	
	/// Computes areas of linear triangles and quads
	void compute_areas();
	
	/** Computes data structures for 
	 * elements surrounding point (esup), 
	 * points surrounding point (psup), 
	 * elements surrounding elements (esuel), 
	 * elements surrounding faces along with points in faces (intfac),
	 * element-face connectivity array elemface (for each facet of each element, 
	 * it stores the intfac face number)
	 * a list of boundary points with correspong global point numbers and containing boundary faces
	 * (according to intfac) (bpoints).
	 * \note
	 * - Use only after setup()
	 * - Currently only works for linear mesh
	 */
	void compute_topological();
	
	/// Computes unit normals and lengths, and sets boundary face tags for all faces in gallfa
	/** \note Uses intfac, so call only after compute_topological, only for linear mesh
	 * \note The normal vector is the UNIT normal vector.
	 * \warning Use only for linear meshes
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
	 * \note Make sure to execute [compute_topological()](@ref compute_topological) 
	 * before calling this function.
	*/
	UMesh2dh convertLinearToQuadratic();

	/// Converts quads in a mesh to triangles
	UMesh2dh convertQuadToTri() const;
};


} // end namespace
#endif

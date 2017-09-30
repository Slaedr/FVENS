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
public:
	UMesh2dh();
	UMesh2dh(const UMesh2dh& other);
	UMesh2dh& operator=(const UMesh2dh& other);
	~UMesh2dh();
		
	/* Functions to get mesh data. */

	/// Returns coordinates of a mesh node
	double gcoords(a_int pointno, int dim) const
	{
		return coords.get(pointno,dim);
	}

	/// Returns global node indices corresponding to local node indices of an element
	a_int ginpoel(a_int elemnum, int localnodenum) const
	{
		return inpoel.get(elemnum, localnodenum);
	}

	/// Returns global node indices or boundary tags corresponding to local node indices of a face
	/** \note The face indexing here could be different from the indexing in the
	 * [face data structure](\ref intfac) \sa gintfac
	 */
	a_int gbface(a_int facenum, int locindex) const
	{
		return bface.get(facenum, locindex);
	}

	a_int gesup(a_int i) const { return esup.get(i); }
	a_int gesup_p(a_int i) const { return esup_p.get(i); }
	a_int gpsup(a_int i) const { return psup.get(i); }
	a_int gpsup_p(a_int i) const { return psup_p.get(i); }

	/// Returns the element adjacent to a given element corresponding to the given local face
	/** Note that the local face number `j' would be the one between local node j and 
	 * local node (j+1) % nfael, where nfael is the total number of faces bounding the given
	 * element.
	 */
	a_int gesuel(a_int ielem, int jface) const { return esuel.get(ielem, jface); }

	/// Returns the face number in the [face data structure](\ref intfac) corresponding to 
	/// the local face index of an element
	a_int gelemface(a_int ielem, int inode) const { return elemface.get(ielem,inode); }

	/// Returns an entry from the face data structure [computed](\ref compute_topological) by us
	a_int gintfac(a_int face, int i) const { return intfac.get(face,i); }

	int gintfacbtags(a_int face, int i) const { return intfacbtags.get(face,i); }
	a_real garea(const a_int ielem) const { return area.get(ielem,0); }
	a_real ggallfa(a_int iface, int index) const { return gallfa.get(iface,index); }
	int gflag_bpoin(const a_int pointno) const { return flag_bpoin.get(pointno); }

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

	/// Set coordinates of a certain point
	/** 'set' counterpart of the 'get' function [gcoords](@ref gcoords).
	 */
	void scoords(const a_int pointno, const int dim, const a_real value)
	{
		coords(pointno,dim) = value;
	}
	
	void modify_bface_marker(int iface, int pos, int number)
	{ bface(iface, pos) = number; }

	/// Reads a mesh file
	/** The file should be in either the Gmsh 2.0 format, the 2D structured Plot3D file
	 * or the rDGFLO Domn format. The file extensions should be
	 * - msh for Gmsh 2.0
	 * - p2d for 2D structured Plot3D
	 * - domn for rDGFLO Domn file.
	 */
	//void readMesh(std::string mfile);

	/// Reads a file in the 2D version of the Plot3D structured format
	void readPlot2d(std::string mfile, const int bci0, const int bcimx, 
			const int bcj0, const int bcjmx);

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
	
	/// Writes out the mesh in the Gmsh 2.0 format
	void writeGmsh2(std::string mfile);

	/// Computes the Jacobian for linear triangles
	void compute_jacobians();

	/// Writes out a list of elements with negative Jacobians; works only for linear triangles
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
	
	/// Elements surrounding elements \sa gesuel
	amat::Array2d<a_int > esuel;
	
	/// Face data structure - contains info about elements and nodes associated with a face
	amat::Array2d<a_int > intfac;
	
	/// Holds boundary tags (markers) corresponding to intfac
	amat::Array2d<int > intfacbtags;
	/// Holds face numbers of faces making up an element
	amat::Array2d<a_int> elemface;
	
	/** \brief Boundary points list
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
};


} // end namespace
#endif

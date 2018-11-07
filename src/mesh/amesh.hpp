/** @file amesh.hpp
 * @brief Contains a class to handle 2D or 3D hybrid meshes
 * @author Aditya Kashi
 */

#ifndef BLASTED_AMESH_H
#define BLASTED_AMESH_H

#include <vector>
#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"

namespace fvens {

/// Hybrid unstructured mesh class supporting various types of cells in both 2D and 3D
template <typename scalar, int ndim>
class UMesh
{
	static_assert(ndim == 2 || ndim == 3, "Only 2D or 3D problems!");

public:
	UMesh();

	~UMesh();
	
	/* Functions to get mesh data are defined right here so as to enable inlining.
	 */

	/// Returns coordinates of a mesh node
	scalar gcoords(const a_int pointno, const int dim) const
	{
		return coords.get(pointno,dim);
	}

	/// Returns global node indices corresponding to local node indices of an element
	a_int gelemnode(const a_int elemnum, const int localnodenum) const
	{
		return inpoel.get(elemnum, localnodenum);
	}

	/// Returns global node indices or boundary tags corresponding to local node indices of a face
	/** \note The face indexing here could be different from the indexing in the
	 * [face data structure](\ref intfac) \sa gintfac
	 */
	a_int gbface(const a_int facenum, const int locindex) const
	{
		return bface.get(facenum, locindex);
	}

	/// Returns elements surrounding points; to be used with \ref gesup_p
	//a_int gesup(const a_int i) const { return esup.get(i); }

	/// Returns the index for \ref gesup to access the list of elems surrounding point i
	//a_int gesup_p(const a_int i) const { return esup_p.get(i); }

	/// Returns the element adjacent to a given element corresponding to the given local face
	/** Note that the local face number `j' would be the one between local node j and
	 * local node (j+1) % nfael, where nfael is the total number of faces bounding the given
	 * element.
	 */
	a_int gesuel(const a_int ielem, const int jface) const { return esuel.get(ielem, jface); }

	/// Returns the face number in the [face data structure](\ref intfac) corresponding to
	/// the local face index of an element
	a_int gelemface(const a_int ielem, const int inode) const { return elemface.get(ielem,inode); }

	/// Returns an entry from the face data structure \ref intfac
	/** \param face Index of the face about which data is needed
	 *   (NOT the same as the index in \ref bface, this is the index in \ref intfac)
	 * \param i An integer which specifies what information is returned:
	 *  - 0: Left cell index
	 *  - 1: Right cell index (or for a boundary face, \ref nelem + face index)
	 *  - 2,3...: Global indices of nodes of the face (in orientation order)
	 */
	a_int gintfac(const a_int face, const int i) const { return intfac.get(face,i); }

	/// Returns the boundary marker of a face indexed by \ref intfac.
	int gintfacbtags(const a_int face, const int i) const { return intfacbtags.get(face,i); }

	/// Returns the measure of a cell
	scalar gvol(const a_int ielem) const { return volume.get(ielem,0); }

	/// Returns the components of the unit normal or the measure of a face \sa facemetric
	scalar gfacemetric(const a_int iface, const int index) const {return facemetric.get(iface,index);}

	/// Returns the unit normal vector as a fixed-size array for a given face of \ref intfac
	std::array<scalar,ndim> gnormal(const a_int iface) const {
		if(ndim == 2)
			return {facemetric.get(iface,0), facemetric.get(iface,1)};
		else
			return {facemetric.get(iface,0), facemetric.get(iface,1), facemetric.get(iface,2)};
	}

	/// Get \ref bface index of a face from its \ref intfac index
	a_int gbifmap(const a_int iface) const { return bifmap(iface); }

	/// Get \ref intfac index of a face from its \ref bface index
	a_int gifbmap(const a_int iface) const { return ifbmap(iface); }

	/// Returns 1 or 0 for a point depending on whether or not it lies on a boundary, respectively
	int gflag_bpoin(const a_int pointno) const { return flag_bpoin.get(pointno); }

	/// Returns the total number of nodes in the mesh
	a_int gnpoin() const { return npoin; }

	/// Returns the total number of elements (cells) in the mesh
	a_int gnelem() const { return nelem; }

	/// Returns the total number of boundary faces in the mesh
	a_int gnface() const { return nface; }

	/// Returns the total number of boundary faces; practically synonymous with \ref gnface
	a_int gnbface() const { return nbface; }

	/// Returns the number of nodes in an element
	int gnnodeElem(const int ielem) const { return nnode[ielem]; }

	/// Returns the total number of faces, both boundary and internal ('Get Number of All FACEs')
	a_int gnaface() const {return naface; }

	/// Returns the number of faces bounding an element
	int gnfaceElem(const int ielem) const { return nfael[ielem]; }

	/// Returns the number of nodes per face in \ref intfac ordering
	int gnnodeFace(const a_int iface) const { return nnofa[iface]; }

	/// Returns the number of boundary tags available for boundary faces
	int gnbtag() const{ return nbtag; }

	/// Returns the number of domain tags available for elements
	int gndtag() const { return ndtag; }

	/// Set coordinates of a certain point
	/** 'set' counterpart of the 'get' function [gcoords](@ref gcoords).
	 */
	void scoords(const a_int pointno, const int dim, const scalar value)
	{
		assert(pointno < npoin);
		assert(dim < NDIM);
		coords(pointno,dim) = value;
	}

	/// Reads a mesh file
	/** The file should be in either the Gmsh 2.0 format, the SU2 format,
	 * or the rDGFLO Domn format. The file extensions should be
	 * - msh for Gmsh 2.0
	 * - su2 for SU2 format
	 * - domn for rDGFLO Domn file.
	 *
	 * \note For an SU2 mesh file, string marker names must be replaced with integers
	 * before this function is called on it.
	 */
	void readMesh(const std::string mfile);

	/// Reads a mesh from a Gmsh 2 format file
	void readGmsh2(const std::string mfile);

	/// Reads a grid in the SU2 format
	void readSU2(const std::string mfile);

	/// Re-orders calls according to some permutation vector
	/** \warning If reordering is needed, this function must be called immediately after reading
	 * the mesh.
	 */
	void reorder_cells(const PetscInt *const permvec);

	/** Stores (in array bpointsb) for each boundary point: the associated global point number
	 * and the two bfaces associated with it.
	 */
	void compute_boundary_points();

	void printmeshstats() const;

	/// Writes out the mesh in the Gmsh 2.0 format
	void writeGmsh2(const std::string mfile);

	/// Computes areas of linear triangles and quads
	void compute_volumes();

	/// Computes locations of cell centres
	/** \param[out] centres Should be logically of size nelem x NDIM.
	 */
	void compute_cell_centres(std::vector<scalar>& centres) const;

	/// Computes some connectivity structures among mesh entities
	/** Computes data structures for
	 * elements surrounding point (esup),
	 * points surrounding point (psup),
	 * elements surrounding elements (esuel),
	 * elements surrounding faces along with points in faces (intfac),
	 * element-face connectivity array elemface (for each facet of each element,
	 * it stores the intfac face number)
	 */
	void compute_topological();

	/// Computes unit normals and lengths, and sets boundary face tags for all faces in intfacbtags
	/** \note Uses intfac, so call only after compute_topological, only for linear mesh
	 * \note The normal vector is the UNIT normal vector.
	 * \sa facemetric
	 * \warning Use only for linear meshes
	 */
	void compute_face_data();

	/// Generates the correspondance between the faces of two periodic boundaries
	/** Sets the indices of ghost cells to corresponding real cells.
	 * \note We assume that there exists precisely one matching face for each face on the
	 *  periodic boundaries, such that their face-centres are aligned.
	 *
	 * \warning Requires \ref compute_topological and \ref compute_face_data to have been called
	 * beforehand, because \ref intfac and \ref intfacbtags is needed.
	 *
	 * \param[in] bcm Marker of one set of periodic boundaries
	 * \param[in] axis The index of the coordinate which is different for the two boundaries
	 *   0 for x, 1 for y, 2 for z. Its the axis along which the geometry is periodic.
	 */
	void compute_periodic_map(const int bcm, const int axis);

	/// Iterates over bfaces and finds the corresponding intfac face for each bface
	/** Stores this data in the boundary label maps \ref ifbmap and \ref bifmap.
	 */
	void compute_boundary_maps();

	/// Writes the boundary point maps [ifbmap](@ref ifbmap) and [bifmap](@ref bifmap) to a file
	void writeBoundaryMapsToFile(std::string mapfile);
	/// Reads the boundary point maps [ifbmap](@ref ifbmap) and [bifmap](@ref bifmap) from a file
	void readBoundaryMapsFromFile(std::string mapfile);

	/// Populate [intfacbtags](@ref intfacbtags) with boundary markers of corresponding bfaces
	void compute_intfacbtags();

	/** \brief Adds high-order nodes to convert a linear mesh to a straight-faced quadratic mesh.
	 *
	 * \note Make sure to execute [compute_topological()](@ref compute_topological)
	 * before calling this function.
	*/
	UMesh convertLinearToQuadratic();

	/// Converts quads in a mesh to triangles
	UMesh convertQuadToTri() const;

private:

	/// Available cell and face types
	enum MEntityType {LINE,TRIANGLE,QUADRANGLE,TETRAHEDRON,PYRAMID,PRISM,HEXAHEDRON};

	/// A generic entity topology that's useless by itself
	/** This is used to refer to an entity topology without knowing its type.
	 */
	struct MGenEntityTopology
	{
		MEntityType type;
		int nVertices;
		int nFaces;
	};

	/// A description of the topology of a specific entity type
	template <int nFace, int maxVerticesPerFace>
	struct MEntityTopology : public MGenEntityTopology
	{
		amat::Array2d<int>> localFaceMap;
	};

	/// A subset of mesh entities having the same type
	struct MEntitySubSet
	{
		/// Beginning index of the subset in the relavant array of entities
		a_int start;
		/// Ending index plus one in the relavant array of entities
		a_int end;
		/// The (unique) topology of entities in this subset
		const MEntityTopology& topo;
	};

	/** \name LFM Local face maps for different types of cells
	 * Given a local face index and the index of a node of that face, they store the index of that
	 * node in the cell.
	 * Note that the ordering here is such that all faces `point' towards the exterior of the cell
	 * by the right-hand rule.
	 */
	///@{
	static const int triLFM[3][2];       ///< Triangle
	static const int quadLFM[4][2];      ///< Quadrangle
	static const int tetLFM[4][3];       ///< Tetrahedron
	static const int pyrLFM[5][4];       ///< Pyramid
	static const int prismLFM[5][4];     ///< (Triangular) Prism
	static const int hexLFM[6][4];       ///< Hexahedron
	///@}

	/** \name elemNnofa Number of nodes in each local face, in the same order as arrays in \ref LFM
	 * However, if all faces contain the same number of nodes, we just store one number.
	 */
	///@{
	static const int triNnofa;
	static const int quadNnofa;
	static const int tetNnofa;
	static const int pyrNnofa[5];
	static const int prismNnofa[5];
	static const int hexNnofa;
	///@}

	a_int npoin;                    ///< Number of nodes
	a_int nelem;                    ///< Number of elements
	a_int nface;                    ///< Number of boundary faces
	std::vector<int> nnode;         ///< number of nodes to an element, for each element
	int maxnnode;                   ///< Maximum number of nodes per element for any element
	std::vector<int> nfael;         ///< number of faces to an element for each element
	int maxnfael;                   ///< Maximum number of faces per element for any element
	int maxnnofa;                   ///< Maximum number of nodes per face for any face
	std::vector<int> nnofa;         ///< number of nodes in a face for each face in \ref intfac
	std::vector<int> nnobfa;        ///< Numbef of nodes in each boundary face, ordered as \ref bface
	a_int naface;                   ///< total number of (internal and boundary) faces
	a_int nbface;                   ///< number of boundary faces as calculated \sa compute_topological
	a_int nbpoin;                   ///< number of boundary points \sa compute_boundary_points
	int nbtag;                      ///< number of tags for each boundary face
	int ndtag;                      ///< number of tags for each element

	/// Coordinates of nodes
	amat::Array2d<scalar> coords;

	/// Interconnectivity matrix: lists node numbers of nodes in each element
	amat::Array2d<a_int > inpoel;

	/// Boundary face data: lists nodes belonging to a boundary face and contains boudnary markers
	amat::Array2d<a_int > bface;

	/// Holds volume region markers, if any
	amat::Array2d<int> vol_regions;

	/// Holds 1 or 0 for each point depending on whether or not that point is a boundary point
	amat::Array2d<int> flag_bpoin;

	/// List of indices of [esup](@ref esup) corresponding to nodes
	amat::Array2d<a_int > esup_p;

	/// List of elements surrounding each point
	/** Integers pointing to particular points' element lists are stored in [esup_p](@ref esup_p).
	 */
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
	/** For details, see \ref gintfac, the accessor function for intfac.
	 */
	amat::Array2d<a_int > intfac;

	/// Holds boundary tags (markers) corresponding to intfac \sa gintfac
	amat::Array2d<int> intfacbtags;

	/// Holds face numbers of faces making up an element
	amat::Array2d<a_int> elemface;

	/// Relates boundary faces in intfac with bface, ie, bifmap(intfac no.) = bface no.
	/** Computed in \ref compute_boundary_maps.
	 */
	amat::Array2d<int> bifmap;

	/// Relates boundary faces in bface with intfac, ie, ifbmap(bface no.) = intfac no.
	/** Computed in \ref compute_boundary_maps.
	 */
	amat::Array2d<int> ifbmap;

	bool isBoundaryMaps;			///< Specifies whether bface-intfac maps have been created

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

	/// Contains volume of each element
	amat::Array2d<scalar> volume;	

	/// Stores lengths and unit normals for linear mesh faces
	/** For each face, the first two/three entries the components of the unit normal,
	 * the third component is the measure of the face.
	 * The unit normal points towards the cell with greater index.
	 */
	amat::Array2d<scalar> facemetric;

	/// Compute lists of elements (cells) surrounding each point \sa esup
	/** \note This function is required to be called before some other topology-related computations.
	 */
	void compute_elementsSurroundingPoints();

	/// Compute lists of elements (cells) surrounding each element \sa esuel
	/** \warning Requires \ref esup and \ref esup_p to be computed beforehand.
	 * \sa compute_elementsSurroundingPoints
	 */
	void compute_elementsSurroundingElements();

	/** \brief Computes, for each face, the elements on either side, the starting node and
	 * the ending node of the face. These are stored in \ref intfac.
	 * Also computes \ref elemface and modifies \ref esuel .
	 *
	 * The orientation of the face is such that the element with smaller index is
	 * always to the left of the face, while the element with greater index
	 * is always to the right of the face.
	 *
	 * The node ordering of the face is such that the face `points' to the cell with greater index;
	 * this means the vector starting at node 0 and pointing towards node 1 would
	 * rotate clockwise by 90 degrees to point to the cell with greater index.
	 *
	 * Also computes element-face connectivity array \ref elemface in the same loop
	 * which computes intfac.
	 *
	 * \note After the following portion, \ref esuel holds (nelem + face no.) for each ghost cell,
	 * instead of -1 as before.
	 */
	void compute_faceConnectivity();

	/// Compute a list of points surrounding each point \sa psup
	void compute_pointsSurroundingPoints();
};

} // end namespace
#endif

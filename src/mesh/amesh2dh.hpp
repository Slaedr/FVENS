/** @file amesh2dh.hpp
 * @brief Contains a class to handle 2D hybrid meshes containing triangles and quadrangles
 * @author Aditya Kashi
 */

#ifndef AMESH2DH_H
#define AMESH2DH_H

#include <vector>
#include "aconstants.hpp"
#include "utilities/aarray2d.hpp"
#include "meshreaders.hpp"

namespace fvens {

class ReplicatedGlobalMeshPartitioner;

/// Hybrid unstructured mesh class supporting triangular and quadrangular elements
template <typename scalar>
class UMesh2dh
{
public:
	UMesh2dh();

	UMesh2dh(const MeshData& md);

	~UMesh2dh();

	/* Functions to get mesh data are defined right here so as to enable inlining.
	 */

	/// Returns coordinates of a mesh node
	scalar gcoords(const a_int pointno, const int dim) const
	{
		return coords.get(pointno,dim);
	}

	/// Returns global node indices corresponding to local node indices of an element
	a_int ginpoel(const a_int elemnum, const int localnodenum) const
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
	a_int gesup(const a_int i) const { return esup.get(i); }

	/// Returns the index for \ref gesup to access the list of elems surrounding point i
	a_int gesup_p(const a_int i) const { return esup_p.get(i); }

	/// Returns points surrounding points; to be used with \ref gpsup_p
	a_int gpsup(const a_int i) const { return psup.get(i); }

	/// Returns the index for \ref gpsup to access the list of points surrounding point i
	a_int gpsup_p(const a_int i) const { return psup_p.get(i); }

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
	 *   Note that boundary faces, connectivity faces or interior faces can be accessed using
	 *   \ref FaceIterators.
	 * \param i An integer which specifies what information is returned:
	 *  - 0: Left cell index
	 *  - 1: Right cell index (or for a boundary face, \ref nelem + face index)
	 *  - 2: Global index of `first' or `starting' node of the face
	 *  - 3: Global index of the `second' or `ending' node of the face
	 */
	a_int gintfac(const a_int face, const int i) const { return intfac.get(face,i); }

	/// \defgroup FaceIterators Iterators over faces indexed according to \ref intfac
	/// @{

	/// Start of physical boundary faces
	a_int gPhyBFaceStart() const;

	/// One index past the end of physical boundary faces
	a_int gPhyBFaceEnd() const;

	/// Start of connection boundary faces (faces connecting this subdomain to other subdomains)
	a_int gConnBFaceStart() const;

	/// One past the end of connection boundary faces
	a_int gConnBFaceEnd() const;

	/// Start of all boundary faces - physical and connectivity
	/** Physical boundary and connectivity faces are stored contiguously.
	 */
	a_int gBFaceStart() const;

	/// One index past the end of all boundary faces
	a_int gBFaceEnd() const;

	/// Start of subdomain faces (all faces other than boundary faces)
	a_int gSubDomFaceStart() const;

	/// One past the end of subdomain faces
	a_int gSubDomFaceEnd() const;

	/// Start of connection boundary and subdomain faces
	/** Beginning of the list of all faces other than physical boundary faces in \ref intfac.
	 * Note that it's guaranteed that conection boundary faces and subdomain faces will be stored
	 * contiguously.
	 */
	a_int gDomFaceStart() const;

	/// One past the end of connection+subdomain faces
	a_int gDomFaceEnd() const;

	/// @}

	/// Returns the boundary marker of a face indexed by \ref intfac.
	int gintfacbtags(const a_int face, const int i) const { return intfacbtags.get(face,i); }

	/// Returns the measure of a cell
	scalar garea(const a_int ielem) const { return area.get(ielem,0); }

	/// Returns the components of the unit normal or the length of a face \sa facemetric
	scalar gfacemetric(const a_int iface, const int index) const {return facemetric.get(iface,index);}

	/// Returns the unit normal vector as a fixed-size array for a given face of \ref intfac
	std::array<scalar,NDIM> gnormal(const a_int iface) const {
#if NDIM == 2
			return {facemetric.get(iface,0), facemetric.get(iface,1)};
#else
			return {facemetric.get(iface,0), facemetric.get(iface,1), facemetric.get(iface,2)};
#endif
	}

	/// Get \ref bface index of a face from its \ref intfac index
	a_int gbifmap(const a_int iface) const { return bifmap(iface); }

	/// Get \ref intfac index of a face from its \ref bface index
	a_int gifbmap(const a_int iface) const { return ifbmap(iface); }

	/// Returns 1 or 0 for a point depending on whether or not it lies on a boundary, respectively
	//int gflag_bpoin(const a_int pointno) const { return flag_bpoin.get(pointno); }

	/// Returns the total number of nodes in the mesh
	a_int gnpoin() const { return npoin; }

	/// Returns the total number of elements (cells) in the mesh
	a_int gnelem() const { return nelem; }

	/// Returns the total number of boundary faces in the mesh
	a_int gnface() const { return nface; }

	/// Returns the total number of boundary faces; practically synonymous with \ref gnface
	a_int gnbface() const { return nbface; }

	/// Returns the number of nodes in an element
	int gnnode(const int ielem) const { return nnode[ielem]; }

	/// Returns the total number of faces, both boundary and internal ('Get Number of All FACEs')
	a_int gnaface() const {return naface; }

	/// Returns the number of connectivity faces (faces adjacent to cells in another subdomain)
	a_int gnConnFace() const { return nconnface; }

	/// Returns the number of faces bounding an element
	int gnfael(const int ielem) const { return nfael[ielem]; }

	/// Returns the number of nodes per face for each face ordered by \ref intfac
	int gnnofa(const int iface) const { return nnofa; }

	/// Returns the number of boundary tags available for boundary faces
	int gnbtag() const{ return nbtag; }

	/// Returns the number of domain tags available for elements
	int gndtag() const { return ndtag; }

	/// Returns the array of global element indices corresponding to external elements across
	///  connectivity boundary faces of this subdomain
	const a_int *getConnectivityGlobalIndices() const
	{
		assert(connGlobalIndices.size() > 0);
		return &connGlobalIndices[0];
	}

	/// Set coordinates of a certain point
	/** 'set' counterpart of the 'get' function [gcoords](@ref gcoords).
	 */
	void scoords(const a_int pointno, const int dim, const scalar value)
	{
		assert(pointno < npoin);
		assert(dim < NDIM);
		coords(pointno,dim) = value;
	}

	/// Re-orders cells according to some permutation vector locally in the subdomain
	/** \warning If reordering is needed, this function must be called immediately after reading
	 * and distributing the mesh.
	 */
	void reorder_cells(const PetscInt *const permvec);

	/** Stores (in array bpointsb) for each boundary point: the associated global point number
	 * and the two bfaces associated with it.
	 */
	void compute_boundary_points();

	void printmeshstats() const;

	/// Writes out the mesh in the Gmsh 2.0 format
	void writeGmsh2(const std::string mfile) const;

	/// Computes areas of linear triangles and quads
	void compute_areas();

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
	 *   0 for x, 1 for y. It's the axis along which the geometry is periodic.
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

	/// The initial mesh partitioner needs direct access to the mesh
	friend UMesh2dh<a_real> partitionMeshTrivial(const MeshData& global_mesh);

	/// Populates this process's share of mesh arrays from the global arrays
	friend void splitMeshArrays(const MeshData& gm, const std::vector<int>& glbElemDist,
	                            UMesh2dh<a_real>& lm);

	friend class ReplicatedGlobalMeshPartitioner;

private:
	// Global properties

	a_int npoinglobal;
	a_int nelemglobal;

	// Local properties

	a_int npoin;                    ///< Number of nodes
	a_int nelem;                    ///< Number of elements
	a_int nface;                    ///< Number of boundary faces
	std::vector<int> nnode;         ///< number of nodes to an element, for each element
	int maxnnode;                   ///< Maximum number of nodes per element for any element
	std::vector<int> nfael;         ///< number of faces to an element for each element
	int maxnfael;                   ///< Maximum number of faces per element for any element
	int nnofa;                      ///< number of nodes in a face
	int nbtag;                      ///< number of tags for each boundary face
	int ndtag;                      ///< number of tags for each element

	/// Coordinates of nodes
	amat::Array2d<scalar> coords;

	/// Interconnectivity matrix: lists node numbers of nodes in each element
	amat::Array2d<a_int> inpoel;

	/// Physical boundary face data
	/// Lists nodes belonging to a boundary face and contains boundary markers
	amat::Array2d<a_int> bface;

	a_int naface;                   ///< total number of (internal and boundary) faces
	/// number of physical boundary faces as calculated in \sa compute_topological
	a_int nbface;
	/// Number of connection boundary faces (connection to other subdomains)
	a_int nconnface;
	a_int nbpoin;                   ///< number of boundary points \sa compute_boundary_points

	/// Connection boundary face data
	/** Contains, for each connectivity boundary face,
	 *   the index of the other subdomain that it connects to,
	 *   the index of the cell in this subdomain that it is a part of, and
	 *   the index of the cell in the other subdomain that it connects to, in that order.
	 */
	amat::Array2d<a_int> connface;

	/// Holds volume region markers, if any
	amat::Array2d<int> vol_regions;

	/// List of indices of [esup](@ref esup) corresponding to nodes
	amat::Array2d<a_int> esup_p;

	/// List of elements surrounding each point
	/** Integers pointing to particular points' element lists are stored in [esup_p](@ref esup_p).
	 */
	amat::Array2d<a_int> esup;

	/// Lists of indices of psup corresponding to nodes (points)
	amat::Array2d<a_int> psup_p;

	/// List of nodes surrounding nodes
	/** Integers pointing to particular nodes' node lists are stored in [psup_p](@ref psup_p)
	 */
	amat::Array2d<a_int> psup;

	/// Elements surrounding elements \sa gesuel
	amat::Array2d<a_int> esuel;

	/// Face data structure - contains info about elements and nodes associated with a face
	/** Currently stores physical boundary faces first, followed by connectivity faces and then
	 * subdomain faces.
	 * For details, see \ref gintfac, the accessor function for intfac.
	 */
	amat::Array2d<a_int> intfac;

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

	/// Contains area of each element (either triangle or quad)
	amat::Array2d<scalar> area;

	/// Stores lengths and unit normals for linear mesh faces
	/** For each face, the first two entries are x- and y-components of the unit normal,
	 * the third component is the length.
	 * The unit normal points towards the cell with greater index.
	 */
	amat::Array2d<scalar> facemetric;

	std::vector<a_int> connGlobalIndices;

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

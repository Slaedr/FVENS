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

/// Index of something w.r.t. the element it is associated with
typedef int EIndex;
/// Index of something (usually a node) w.r.t. the face it is associated with
typedef int FIndex;

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

	/// Access to physical boundary face information in this subdomain (if any)
	/** Returns global node indices or boundary tags corresponding to local node indices of a face
	 * \note The face indexing here could be different from the indexing in the
	 * [face data structure](\ref intfac) \sa gintfac
	 */
	a_int gbface(const a_int facenum, const int locindex) const
	{
		return bface.get(facenum, locindex);
	}

	/// Access to the connectivity boundary face information in case of multiprocess runs
	/** \param[in] icface Connectivity face index in arbirtray order
	 * \param[in] infoindex To query information about the face:
	 *  - 0: Subdomain index of the element this face belongs to
	 *  - 1: The EIndex of the face in the element it belongs to
	 *  - 2: The subdomain rank of the other adjacent element which is external (to this subdomain)
	 *  - 3: The global element index of the external neighboring element
	 */
	a_int gconnface(const a_int icface, const int infoindex) const
	{ return connface(icface, infoindex); }

	/// Returns elements surrounding points; to be used with \ref gesup_p
	a_int gesup(const a_int i) const { return esup.get(i); }

	/// Returns the index for \ref gesup to access the list of elems surrounding point i
	a_int gesup_p(const a_int i) const { return esup_p.get(i); }

	/// Returns points surrounding points; to be used with \ref gpsup_p
	a_int gpsup(const a_int i) const { return psup.get(i); }

	/// Returns the index for \ref gpsup to access the list of points surrounding point i
	a_int gpsup_p(const a_int i) const { return psup_p.get(i); }

	/// Returns the element adjacent to a given element corresponding to the given local face
	/** Note that in 2D, the local face number `j' would be the one between local node j and
	 * local node (j+1) % nfael, where nfael is the total number of faces bounding the given
	 * element.
	 *
	 * The connectivity ghost cell opposite the intfac face at index iface is assigned the index
	 * nelem+iface.
	 * The physical boundary ghost cell opposite the intfac face at index iface is assigned the index
	 * nelem+nconnface+iface.
	 * This function returns the correct element index for respective faces.
	 * This assignment is done by \ref compute_faceConnectivity or \ref compute_topological.
	 */
	a_int gesuel(const a_int ielem, const int jface) const { return esuel.get(ielem, jface); }

	/// Returns the face number in the [face data structure](\ref intfac) corresponding to
	/// the local face index of an element
	/** For a connectivity ghost cell, only elemface(ielem,0) is defined.
	 */
	a_int gelemface(const a_int ielem, const EIndex ifael) const { return elemface.get(ielem,ifael); }

	/// Returns the global element index of an element of this subdomain
	a_int gglobalElemIndex(const a_int iel) const { return globalElemIndex[iel]; }

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
	a_int gPhyBFaceStart() const { return phyBFaceStart; }

	/// One index past the end of physical boundary faces
	a_int gPhyBFaceEnd() const { return phyBFaceEnd; }

	/// Start of connection boundary faces (faces connecting this subdomain to other subdomains)
	a_int gConnBFaceStart() const { return connBFaceStart; }

	/// One past the end of connection boundary faces
	a_int gConnBFaceEnd() const { return connBFaceEnd; }

	/// Start of subdomain faces (faces in the interior of the subdomain)
	a_int gSubDomFaceStart() const { return subDomFaceStart; }

	/// One past the end of subdomain faces
	a_int gSubDomFaceEnd() const { return subDomFaceEnd; }

	/// Start of connection boundary and interior subdomain faces
	/** Beginning of the list of all faces other than physical boundary faces in \ref intfac.
	 * Note that it's guaranteed that conection boundary faces and subdomain faces will be stored
	 * contiguously.
	 */
	a_int gDomFaceStart() const { return domFaceStart; }

	/// One past the end of connection+subdomain faces
	a_int gDomFaceEnd() const { return domFaceEnd; }

	/// Beginning of the list of all faces
	a_int gFaceStart() const { return 0; }

	/// One past the end of the list of all faces
	a_int gFaceEnd() const { return naface; }

	/// @}

	/// Returns the boundary marker of a boundary face indexed by \ref intfac.
	/** Note that the index passed here must be in the range gPhyBFaceStart to gPhyBFaceEnd.
	 */
	int gbtags(const a_int face, const int i) const { return btags.get(face-gPhyBFaceStart(),i); }

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

	/// Returns 1 or 0 for a point depending on whether or not it lies on a boundary, respectively
	//int gflag_bpoin(const a_int pointno) const { return flag_bpoin.get(pointno); }

	/// Returns the total number of elements in the entire distributed mesh
	a_int gnelemglobal() const { return nelemglobal; }

	/// Returns the total number of vertices in the entire distributed mesh
	a_int gnpoinglobal() const { return npoinglobal; }

	/// Returns the total number of nodes in the mesh
	a_int gnpoin() const { return npoin; }

	/// Returns the total number of elements (cells) in the mesh
	a_int gnelem() const { return nelem; }

	/// Returns the total number of physical boundary faces
	a_int gnbface() const { return nbface; }

	/// Returns the number of nodes in an element
	int gnnode(const int ielem) const { return nnode[ielem]; }

	/// Returns the total number of faces, both boundary and internal ('Get Number of All FACEs')
	a_int gnaface() const {return naface; }

	/// Number of interior faces in the mesh subdomain
	a_int gninface() const { return ninface; }

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
	// const a_int *getConnectivityGlobalIndices() const
	// {
	// 	assert(connGlobalIndices.size() > 0);
	// 	return &connGlobalIndices[0];
	// }
	std::vector<a_int> getConnectivityGlobalIndices() const;

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
	/** \param[in,out] centres Should be logically of size nelem x NDIM. Contains cell-centre coords
	 *    on output.
	 */
	void compute_cell_centres(scalar *const centres) const;

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

	/// Computes unit normals and lengths, and sets boundary face tags for all faces in btags
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
	 * beforehand, because \ref intfac and \ref btags is needed.
	 *
	 * \param[in] bcm Marker of one set of periodic boundaries
	 * \param[in] axis The index of the coordinate which is different for the two boundaries
	 *   0 for x, 1 for y. It's the axis along which the geometry is periodic.
	 */
	void compute_periodic_map(const int bcm, const int axis);

	/// Get the index of a node w.r.t. an element (ie., get the node's "EIndex") from
	///  its index in a face of that element
	/** \param ielem Element index in the subdomain
	 * \param faceEIndex Index of a face in the element w.r.t. the element (ie., the face's EIndex)
	 * \param nodeFIndex Index of a node in the face above w.r.t. the face (ie., the node's FIndex)
	 *
	 * The current implementation works only in 2D.
	 */
	constexpr EIndex getNodeEIndex(const a_int ielem, const EIndex iface, const FIndex inode) const {
		static_assert(NDIM==2);
		return (iface + inode) % nnode[ielem];
	}

	/// Returns the EIndex of a face in a certain element
	/** Returns negative if the face is not present in that element.
	 * \warning If iface is a physical boundary face, this function will always work. But if iface is
	 *  an interior face defined according to \ref intfac, obviously intfac must be available.
	 * \param[in] phyboundary If true, iface is interpreted as a bface index between 0 and nbface.
	 *   If false, iface is interpreted as an intfac index.
	 */
	EIndex getFaceEIndex(const bool phyboundary, const a_int iface, const a_int elem) const;

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
	a_int nbface;                    ///< Number of boundary faces
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

	/// Holds volume region markers, if any
	amat::Array2d<int> vol_regions;

	a_int naface;                   ///< total number of (internal and boundary) faces
	a_int ninface;                  ///< Total number of interior faces
	/// Number of connection boundary faces (connection to other subdomains)
	a_int nconnface;
	a_int nbpoin;                   ///< number of boundary points \sa compute_boundary_points

	a_int connBFaceStart;           ///< Starting index of list of connectivity faces
	a_int connBFaceEnd;             ///< One past the end of the list of connectivity faces
	a_int subDomFaceStart;          ///< Start of list of interior faces of this subdomain
	a_int subDomFaceEnd;            ///< One past of the end of the list of interior faces
	a_int domFaceStart;             ///< Start of list of connectivity and subdomain faces
	a_int domFaceEnd;               ///< One past the end of the list of connectivity and subdomain faces
	a_int phyBFaceStart;            ///< Start of the list of physical boundary faces
	a_int phyBFaceEnd;              ///< One past the end of the list of physical boundary faces

	/// Connection boundary face data
	/** Contains, for each connectivity boundary face,
	 *   the index of the cell in this subdomain that it is a part of,
	 *   the local face EIndex of the boundary face in that cell
	 *   the index of the other subdomain that it connects to, and
	 *   the global index of the cell in the other subdomain that it connects to, in that order.
	 */
	amat::Array2d<a_int> connface;

	/// Stores global element indices of each element in this subdomain
	std::vector<a_int> globalElemIndex;

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
	amat::Array2d<int> btags;

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

	/// Computes the interior element associated with each physical boundary face of this subdomain
	/** \return A pair with the interior element index as the first entry and
	 *    the EIndex of the face in that element as the second entry.
	 */
	std::vector<std::pair<a_int,EIndex>> compute_phyBFaceNeighboringElements() const;

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

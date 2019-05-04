/** \file ameshutils.hpp
 * \brief Some mesh-related functionality
 * \author Aditya Kashi
 * \date 2018-02
 */

#ifndef AMESHUTILS_H
#define AMESHUTILS_H

#include "mesh.hpp"
#include "spatial/aspatial.hpp"

namespace fvens {

/// Returns a ready-to-use mesh object from the path to mesh file
UMesh<freal,2> constructMesh(const std::string mesh_path);

/// Computes various entity lists required for mesh traversal, also reorders the cells if requested
/** This can, and should, be called immediately after [reading](UMesh2dh::readMesh) the mesh.
 * Does not compute [periodic boundary maps](UMesh2dh::compute_periodic_map);
 * this must be done separately.
 */
template <typename scalar>
StatusCode preprocessMesh(UMesh<scalar,2>& m);

/// Divides mesh cells into levels within each of which no cell is coupled to another
/** Returns a list of cell indices corresponding to the start of each level.
 * The length of the list is one more than the number of levels.
 */
template <typename scalar>
std::vector<fint> levelSchedule(const UMesh<scalar,2>& m);

/// Compares two meshes for equality
/** \return An array of booleans which represents whether the following, in order, are the same:
 *  - number of elements
 *  - number of points
 *  - number of physical boundary faces
 *  - number of nodes per element for each element
 *  - number of faces per element for each element
 *  - element-point interconnectivity list
 *  - boundary faces' array along with boundary tags
 *  - coordinates of points
 */
std::array<bool,8> compareMeshes(const UMesh<freal,2>& m1, const UMesh<freal,2>& m2);

/// Computes cell-centre coordinates for one cell. Assumes an array of structures.
template <typename scalar> inline
void kernel_computeCellCentreAoS(const UMesh<scalar,2>& m, const fint cellidx,
                                 scalar *const ccentres)
{
	for(int i = 0; i < NDIM; i++)
		ccentres[i] = 0;

	for(EIndex j = 0; j < m.gnnode(cellidx); j++)
		for(int k = 0; k < NDIM; k++)
			ccentres[k] += m.gcoords(m.ginpoel(cellidx,j),k);

	for(int i = 0; i < NDIM; i++)
		ccentres[i] /= m.gnnode(cellidx);
}

}
#endif

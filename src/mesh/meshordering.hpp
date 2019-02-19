/** \file
 * \brief Some useful orderings for cells in the mesh
 * \author Aditya Kashi
 * \date 2019-02
 */

#ifndef FVENS_MESH_ORDERING_H
#define FVENS_MESH_ORDERING_H

#include "amesh2dh.hpp"

namespace fvens {

/// Computes an ordering where lines of strong coupling are identified and ordered consecutively
/** The outline of the algorithm is taken from \cite mavriplis_anisotropic. However, that paper is
 * is ambiguous about the local anisotropy metric. We use the max face weight divided by
 * the min face weight at each cell.
 * \param m The mesh to be reordered (it is assumed that 'elements surrounding elements' is available)
 * \param threshold The lower limit for the local anisotropy metric for which lines will be extended
 */
template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const a_real threshold);


struct LineConfig {
	/// Indices of cells that make up lines
	std::vector<std::vector<a_int>> lines;
	/// For each cell, stores the line number it belongs to, if applicable, otherwise stores -1
	std::vector<int> celline;
};

/// Finds lines in the mesh
template <typename scalar>
LineConfig findLines(const UMesh2dh<scalar>& m, const a_real threshold);

}

#endif

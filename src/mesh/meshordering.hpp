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
 * is slightly ambiguous about the local anisotropy metric. We use the max face weight divided by
 * the min face weight at each cell.
 * \param m The mesh to be reordered (it is assumed that 'elements surrounding elements' is available)
 * \param threshold The lower limit for the local anisotropy metric for which lines will be extended
 */
template <typename scalar>
void lineReorder(UMesh2dh<scalar>& m, const a_real threshold);

/// Orders the mesh cells according to a hybrid of line ordering and some other specified ordering.
/** Divides the mesh cells into lines and points, determines the connectivity between lines and points
 *  and orders the lines and points according to some desired ordering. Cells within a line are
 *  ordered consecutively.
 * \param m The mesh to be reordered
 * \param threshold Local anisotropy threshold
 * \param ordering The ordering to use for the lines and points
 */
template <typename scalar>
void hybridLineReorder(UMesh2dh<scalar>& m, const a_real threshold, const char *const ordering);

}

#endif
